#!/bin/bash

./run tc_gan/run/moments.py -- --gen-learn-rate=.001 --IO_type=asym_tanh --N=201 --n_bandwidths=5 --rate_cost=0 --truth_size=1000 --sample-sites=0,.2,.4,.6 --track_offset_identity --truth_size_per_batch=1000 --lam=.1 --contrast=5,20 --init-disturbance=0 --n_samples=25

./run tc_gan/run/moments.py -- --gen-learn-rate=.001 --IO_type=asym_tanh --N=201 --n_bandwidths=5 --rate_cost=0 --truth_size=1000 --sample-sites=0,.2,.4,.6 --track_offset_identity --truth_size_per_batch=1000 --lam=.1 --contrast=5,20 --init-disturbance=0 --n_samples=50

./run tc_gan/run/moments.py -- --gen-learn-rate=.001 --IO_type=asym_tanh --N=201 --n_bandwidths=5 --rate_cost=0 --truth_size=1000 --sample-sites=0,.2,.4,.6 --track_offset_identity --truth_size_per_batch=1000 --lam=.1 --contrast=5,20 --init-disturbance=0 --n_samples=100

./run tc_gan/run/moments.py -- --gen-learn-rate=.001 --IO_type=asym_tanh --N=201 --n_bandwidths=5 --rate_cost=0 --truth_size=1000 --sample-sites=0,.2,.4,.6 --track_offset_identity --truth_size_per_batch=1000 --lam=0 --contrast=5,20 --init-disturbance=-.1 --n_samples=100

