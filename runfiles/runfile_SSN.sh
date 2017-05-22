#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

./python run_batch_GAN.py --layers [512,512] --n_samples 30 --gen-learn-rate .01 --disc-learn-rate .01 --WGAN --IO_type asym_tanh --N 201 --WGAN_lambda 10 --n_bandwidths 8 --rate_cost 1000