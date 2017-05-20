#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

./python run_batch_GAN.py --layers [128] --n_samples 15 --gen-learn-rate .0001 --disc-learn-rate .0001 --WGAN --IO_type asym_power --N 201 --WGAN_lambda 50 --n_bandwidths 8