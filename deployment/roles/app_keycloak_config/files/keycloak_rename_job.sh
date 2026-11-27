#!/bin/bash

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Renames the Keycloak objects of a release before the Chatbot Toolkit rename in
# place, keeping their ids, so passwords, role mappings, group memberships and the
# token subject stay. Each object is renamed only while its old name exists and its
# new one does not, and the realm last, so a run that stopped part way resumes.

set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:?}"
KEYCLOAK_USER="${KEYCLOAK_ADMIN_USER:?}"
ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:?}"
OLD_REALM="${KEYCLOAK_PREVIOUS_REALM:?}"
NEW_REALM="${KEYCLOAK_REALM:?}"
SECRET_NAMESPACE="${SECRET_NAMESPACE:?}"
# The Job's Role grants these names only (templates/rename-job.yaml.j2).
readonly OLD_SECRET_NAME=erag-credentials NEW_SECRET_NAME=chattool-credentials
# Holds the realm's editUsernameAllowed while it is switched on to rename users,
# so a run that stopped before switching it back restores the original value.
readonly STATE_CONFIGMAP=keycloak-rename-state
readonly LIST_MAX=1000

if [ -n "${KUBERNETES_SERVICE_HOST:-}" ] && [ -n "${no_proxy:-}" ]; then
  export no_proxy="${no_proxy},${KUBERNETES_SERVICE_HOST}"
fi

BODY_FILE=$(mktemp)
# The bearer token, as a curl config file: a token refreshed inside $(get ...) reaches
# the later requests, and it never appears in the process arguments.
TOKEN_CONFIG=$(umask 077; mktemp)
USERNAME_EDITS_ON=false
# A run that fails while usernames are editable switches them back; the state
# ConfigMap covers a run that is killed.
# The restore runs in a subshell, so the token file is removed even when it fails.
trap 'if [ "$USERNAME_EDITS_ON" = true ]; then (restore_username_edits) || true; fi; rm -f "$BODY_FILE" "$TOKEN_CONFIG"' EXIT

log_info() { echo "[INFO] $1"; }
log_error() { echo "[ERROR] $1" >&2; }

get_access_token() {
  local response
  response=$(printf '%s' "$ADMIN_PASSWORD" | curl -sS --fail-with-body -X POST \
    "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
    --data-urlencode "username=${KEYCLOAK_USER}" \
    --data-urlencode 'password@-' \
    --data-urlencode 'grant_type=password' \
    --data-urlencode 'client_id=admin-cli') || { log_error "No admin token from ${KEYCLOAK_URL}"; exit 1; }
  local token
  token=$(jq -r '.access_token // empty' <<< "$response")
  [ -n "$token" ] || { log_error "No admin token from ${KEYCLOAK_URL}"; exit 1; }
  printf 'header = "Authorization: Bearer %s"\n' "$token" > "$TOKEN_CONFIG"
}

api() {
  local method=$1 path=$2 json=${3:-} attempt
  for attempt in 1 2; do
    local args=(-sS -o "$BODY_FILE" -w '%{http_code}' -X "$method" "${KEYCLOAK_URL}/admin/realms${path}"
      -K "$TOKEN_CONFIG" -H 'Content-Type: application/json')
    [ -n "$json" ] && args+=(--data-binary "$json")
    HTTP_CODE=$(curl "${args[@]}") || HTTP_CODE=000
    [ "$HTTP_CODE" = 401 ] && [ "$attempt" = 1 ] && { get_access_token; continue; }
    break
  done
}

get() {
  api GET "$1"
  [[ "$HTTP_CODE" =~ ^2 ]] || { log_error "GET $1 returned HTTP $HTTP_CODE: $(head -c 300 "$BODY_FILE")"; exit 1; }
  cat "$BODY_FILE"
}

call() {
  api "$@"
  [[ "$HTTP_CODE" =~ ^2 ]] || { log_error "$1 $2 returned HTTP $HTTP_CODE: $(head -c 300 "$BODY_FILE")"; exit 1; }
}

JQ_RENAME='def rn: gsub("EnterpriseRAG"; "ChatbotToolkit") | gsub("\\bERAG-"; "CHATTOOL-") | gsub("\\berag-"; "chattool-");'
new_name() { jq -rn --arg s "$1" "${JQ_RENAME} \$s | rn"; }

realm_exists() {
  api GET "/$1"
  case "$HTTP_CODE" in
    200) return 0 ;;
    404) return 1 ;;
    *) log_error "Looking up realm $1 returned HTTP $HTTP_CODE"; exit 1 ;;
  esac
}

rename_objects() {
  local what=$1 list_path=$2 item_prefix=$3 field=$4 filter=$5
  local listing selected old new id
  listing=$(get "$list_path")
  # Capped listings: a rename left out would never be retried once the realm is renamed.
  if [[ "$list_path" == *"max=${LIST_MAX}"* ]] && [ "$(jq 'length' <<< "$listing")" -ge "$LIST_MAX" ]; then
    log_error "$what listing reached $LIST_MAX entries; rename them by hand, then run again"
    exit 1
  fi
  selected=$(jq -r --arg f "$field" ".[] | select(${filter}) | [.id, .[\$f]] | @tsv" <<< "$listing")
  while IFS=$'\t' read -r id old; do
    [ -n "$id" ] || continue
    new=$(new_name "$old")
    if name_taken "$item_prefix" "$listing" "$field" "$new"; then
      log_info "$what '$new' exists - leaving '$old' as it is"
      continue
    fi
    local object
    object=$(get "${item_prefix}/${id}")
    call PUT "${item_prefix}/${id}" "$(jq -c --arg f "$field" --arg n "$new" '.[$f] = $n' <<< "$object")"
    log_info "$what '$old' renamed to '$new'"
  done <<< "$selected"
}

# A user listing is a search for the old prefix, so it does not show the new name.
name_taken() {
  local item_prefix=$1 listing=$2 field=$3 name=$4 found
  if [ "$field" = username ]; then
    # Called as an if condition, where set -e does not stop the run.
    found=$(get "${item_prefix}?exact=true&username=$(jq -rn --arg u "$name" '$u | @uri')") || exit 1
    [ "$(jq 'length' <<< "$found")" != 0 ]
  else
    jq -e --arg f "$field" --arg n "$name" 'any(.[]; .[$f] == $n)' <<< "$listing" >/dev/null
  fi
}

# Mapper names, and the settings that name a client, a role or a group, under the
# new names; other settings, such as a directory DN, are the site's own. Of the
# directory group filters only the three defaults follow, as the docs ask for those
# groups to be renamed in the directory; a filter the site set is its own.
JQ_MAPPER="${JQ_RENAME}"' def rnfilter: if test("^\\(cn=erag-(admin|user|maintainer)-group\\)$") then rn else . end;
  def each(f): if type == "array" then map(f) elif type == "string" then f else . end;
  def rnmapper: .name |= rn
  | if .config then .config |= with_entries(
      if (.key | IN("included.client.audience", "usermodel.clientRoleMapping.clientId", "claim.value", "role",
                    "group", "client.id"))
      then .value |= each(rn)
      elif (.key | IN("roles.ldap.filter", "groups.ldap.filter")) then .value |= each(rnfilter)
      else . end)
    else . end;'

rename_mappers() {
  local what=$1 list_path=$2 item_prefix=${3:-$2} listing changed pair old_name mapper id
  listing=$(get "$list_path")
  changed=$(jq -c "${JQ_MAPPER}"' .[] | select(. != rnmapper) | [.name, rnmapper]' <<< "$listing")
  while IFS= read -r pair; do
    [ -n "$pair" ] || continue
    old_name=$(jq -r '.[0]' <<< "$pair")
    mapper=$(jq -c '.[1]' <<< "$pair")
    id=$(jq -r '.id' <<< "$mapper")
    if [ "$old_name" = "$(jq -r '.name' <<< "$mapper")" ]; then
      call PUT "${item_prefix}/${id}" "$mapper"
      log_info "$what '$old_name' now uses the new names"
      continue
    fi
    # Keycloak keeps a mapper's name on update, so a renamed one is created anew.
    # One a stopped run, or a configurator, created under the new name is kept.
    if ! jq -e --arg n "$(jq -r '.name' <<< "$mapper")" 'any(.[]; .name == $n)' <<< "$listing" >/dev/null; then
      call POST "$item_prefix" "$(jq -c 'del(.id)' <<< "$mapper")"
    fi
    call DELETE "${item_prefix}/${id}"
    log_info "$what '$old_name' replaced by '$(jq -r '.name' <<< "$mapper")'"
  done <<< "$changed"
}

allow_username_edits() {
  local original
  if kubectl get configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" >/dev/null 2>&1; then
    log_info "Resuming a rename that stopped part way"
  else
    original=$(get "/${OLD_REALM}" | jq -r '.editUsernameAllowed // false')
    kubectl create configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" \
      --from-literal=editUsernameAllowed="$original" >/dev/null
  fi
  USERNAME_EDITS_ON=true
  call PUT "/${OLD_REALM}" '{"editUsernameAllowed": true}'
}

restore_username_edits() {
  local original
  original=$(kubectl get configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" -o jsonpath='{.data.editUsernameAllowed}')
  [[ "$original" == true || "$original" == false ]] || { log_error "Unexpected editUsernameAllowed '$original' in $STATE_CONFIGMAP"; exit 1; }
  USERNAME_EDITS_ON=false
  call PUT "/${OLD_REALM}" "{\"editUsernameAllowed\": ${original}}"
}

# The configurator reads the generated passwords from the new Secret, so it gets
# the old one's under the new key names; usernames follow the renamed users. The
# old Secret is left in place.
copy_credentials() {
  if kubectl get secret "$NEW_SECRET_NAME" -n "$SECRET_NAMESPACE" >/dev/null 2>&1; then
    log_info "Secret $NEW_SECRET_NAME exists - leaving it as it is"
    return 0
  fi
  local old
  # The Job runs only while the old Secret exists, so any failure here is an error.
  old=$(kubectl get secret "$OLD_SECRET_NAME" -n "$SECRET_NAMESPACE" -o json) \
    || { log_error "Cannot read Secret $OLD_SECRET_NAME in namespace $SECRET_NAMESPACE"; exit 1; }
  jq --arg name "$NEW_SECRET_NAME" "${JQ_RENAME}"' {
        apiVersion: "v1", kind: "Secret", type: .type,
        metadata: {name: $name, namespace: .metadata.namespace,
          labels: (.metadata.labels // {} | with_entries(select(.key | startswith("velero.io/") | not)))},
        data: (.data | with_entries(.key |= sub("^KEYCLOAK_ERAG_"; "KEYCLOAK_CHATTOOL_")
          | if (.key | endswith("_USERNAME")) then .value |= (@base64d | rn | @base64) else . end))}' <<< "$old" \
    | kubectl create -f - >/dev/null
  log_info "Secret $OLD_SECRET_NAME copied to $NEW_SECRET_NAME under the new key names"
}

# Keycloak may still be restarting after an upgrade, and this Job is not retried.
for attempt in $(seq 1 60); do
  curl -sf --max-time 5 -o /dev/null "${KEYCLOAK_URL}/realms/master" && break
  [ "$attempt" -lt 60 ] || { log_error "Keycloak at ${KEYCLOAK_URL} not ready after 5 minutes"; exit 1; }
  sleep 5
done
get_access_token

if ! realm_exists "$OLD_REALM"; then
  # Only the release's own realm name is renamed; one set in the configuration is kept.
  if [ "$OLD_REALM" != EnterpriseRAG ] && realm_exists EnterpriseRAG; then
    log_error "Realm EnterpriseRAG of the earlier release exists, but the configuration names realm $OLD_REALM. Set chattool_keycloak_realm to the earlier release's realm name, or remove it to use ChatbotToolkit, then run again."
    exit 1
  fi
  log_info "No realm $OLD_REALM - nothing to rename"
  kubectl delete configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" --ignore-not-found >/dev/null
  exit 0
fi
# A realm that keeps its name is done once its credentials are copied, the last
# rename step; the old Secret, which starts this Job, stays until teardown.
if [ "$OLD_REALM" = "$NEW_REALM" ] && kubectl get secret "$NEW_SECRET_NAME" -n "$SECRET_NAMESPACE" >/dev/null 2>&1; then
  # Clients are renamed first, so an old one left means a configurator of the new
  # release ran over the old objects and created a second set beside them.
  clients=$(get "/${OLD_REALM}/clients")
  if jq -e 'any(.[]; .clientId | startswith("EnterpriseRAG-"))' <<< "$clients" >/dev/null; then
    log_error "Realm $OLD_REALM holds clients under both the old and the new names. Delete the ChatbotToolkit-* clients, the CHATTOOL-* roles, the chattool-* users and groups, and the Secret $NEW_SECRET_NAME in namespace $SECRET_NAMESPACE, then run again."
    exit 1
  fi
  log_info "Realm $OLD_REALM was already renamed inside - nothing to rename"
  kubectl delete configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" --ignore-not-found >/dev/null
  exit 0
fi
if [ "$OLD_REALM" != "$NEW_REALM" ] && realm_exists "$NEW_REALM"; then
  log_error "Both realms $OLD_REALM and $NEW_REALM exist. Keep the one holding the users and delete the other. If you keep $OLD_REALM, also delete the Secret $NEW_SECRET_NAME in namespace $SECRET_NAMESPACE. Then run again."
  exit 1
fi

R="/${OLD_REALM}"
rename_objects "Client" "${R}/clients" "${R}/clients" clientId '.clientId | startswith("EnterpriseRAG-")'
clients=$(get "${R}/clients")
for client_id in $(jq -r '.[] | select(.clientId | startswith("ChatbotToolkit-")) | .id' <<< "$clients"); do
  rename_objects "Client role" "${R}/clients/${client_id}/roles?briefRepresentation=false" "${R}/roles-by-id" name \
    '.name | test("^(ERAG|erag)-")'
done
rename_objects "Realm role" "${R}/roles?briefRepresentation=false" "${R}/roles-by-id" name '.name | startswith("ERAG-")'
rename_objects "Group" "${R}/groups?max=${LIST_MAX}" "${R}/groups" name '.name | startswith("erag-")'

allow_username_edits
# Only the users the configurator creates: other names, and the users of a
# directory, are not the release's to change.
rename_objects "User" "${R}/users?username=erag-&max=${LIST_MAX}&briefRepresentation=false" "${R}/users" username \
  '(.username | IN("erag-admin", "erag-user", "erag-maintainer")) and .federationLink == null'
restore_username_edits

for client_id in $(jq -r '.[].id' <<< "$clients"); do
  rename_mappers "Client mapper" "${R}/clients/${client_id}/protocol-mappers/models"
done
providers=$(get "${R}/identity-provider/instances")
for alias in $(jq -r '.[].alias' <<< "$providers"); do
  rename_mappers "Identity provider mapper" "${R}/identity-provider/instances/${alias}/mappers"
done
# Directory mappers are listed under their directory, not under the realm.
directories=$(get "${R}/components?type=org.keycloak.storage.UserStorageProvider")
for directory_id in $(jq -r '.[].id' <<< "$directories"); do
  rename_mappers "Directory mapper" \
    "${R}/components?parent=${directory_id}&type=org.keycloak.storage.ldap.mappers.LDAPStorageMapper" "${R}/components"
done

copy_credentials

if [ "$OLD_REALM" != "$NEW_REALM" ]; then
  call PUT "$R" "{\"realm\": \"${NEW_REALM}\"}"
  log_info "Realm $OLD_REALM renamed to $NEW_REALM"
fi
kubectl delete configmap "$STATE_CONFIGMAP" -n "$SECRET_NAMESPACE" --ignore-not-found >/dev/null
