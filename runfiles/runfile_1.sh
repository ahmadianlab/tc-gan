#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

./python run_batch_GAN.py --loss CE --layers [128] --n_samples 30 --IO_type asym_tanh 