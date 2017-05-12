#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

python run_batch_GAN.py --loss "$1" --layers "$2" --n_samples "$3" --IO_type "$4" --rate_cost "$5"