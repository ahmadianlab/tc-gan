#!/bin/bash
thisdir="$(dirname "${BASH_SOURCE[0]}")"
RUN="${1:-../../../../run}"
cd "$thisdir"
exec "$RUN" -- tc_gan.run.bptt_cwgan --load-config run.json --datastore .
