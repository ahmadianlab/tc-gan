#!/bin/bash
RUN="${1:-/opt/tc-gan/run}"
exec "$RUN" -- tc_gan.run.bptt_moments --load-config run.json --datastore .
