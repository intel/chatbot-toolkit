#!/bin/sh
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Without CNI chaining, as on OpenShift, ztunnel captures a pod only once one of
# its containers runs; a connection opened before that leaves as plain TCP and is
# dropped. Captured, the pod listens on HBONE 15008 (3AA0 in /proc/net/tcp*).
# Shell builtins only: the query images do not all ship grep.
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
