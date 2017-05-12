#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

python run_batch_GAN.py --loss CE --layers [128,128]