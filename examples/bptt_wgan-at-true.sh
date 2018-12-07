#!/bin/sh
exec ./run "$@" -- tc_gan.run.bptt_wgan \
     --load-config examples/params/bptt_wgan-at-true.yaml \
     --debug
