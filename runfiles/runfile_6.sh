#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

python run_batch_GAN.py --loss LS --layers [128,128] --n_samples 20 --IO_type asym_linear