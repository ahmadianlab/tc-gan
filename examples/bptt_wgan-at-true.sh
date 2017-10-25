#!/bin/sh
exec ./run "$@" -- nips_madness.run.bptt_wgan \
     --load-config examples/params/bptt_wgan-at-true.yaml \
     --debug
