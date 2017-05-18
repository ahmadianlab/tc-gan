#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

./python run_batch_GAN.py --loss CE --layers [128,128] --n_samples 15 --gen-learn-rate .001 --disc-learn-rate .001 --IO_type asym_power --N 200