#!/bin/sh
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Waits until ztunnel has taken over the pod's network, so the query's first
# connection goes through the mesh. Without CNI chaining, as on OpenShift,
# ztunnel captures a pod only once one of its containers runs, and a connection
# opened before that leaves as plain TCP that the STRICT mesh and the
# namespace's NetworkPolicy drop. Where istio-cni is chained, the pod is
# captured before it starts and this returns at once. Captured, the pod
# listens on ztunnel's HBONE port 15008 (3AA0 in /proc/net/tcp*). Only shell
# builtins and sleep are used: the query images do not all ship grep.
set -u

waited=0
while [ "$waited" -lt "$ZTUNNEL_WAIT_SECONDS" ]; do
  for table in /proc/net/tcp6 /proc/net/tcp; do
    [ -r "$table" ] || continue
    while read -r _ local_address _ state _; do
      case "$local_address" in
        *:3AA0) [ "$state" = "0A" ] && exit 0 ;;
      esac
    done < "$table"
  done
  sleep 1
  waited=$((waited + 1))
done

echo "ztunnel did not capture the pod within ${ZTUNNEL_WAIT_SECONDS}s; querying anyway" >&2
