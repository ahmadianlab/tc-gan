#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

./python run_batch_GAN.py --loss CE --layers [128] --n_samples 30 --rate_cost 1 --gen-learn-rate .001 --disc-learn-rate .001 --WGAN