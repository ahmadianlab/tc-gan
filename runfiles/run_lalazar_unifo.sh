#!/bin/bash

cd research/NIPS_madness

module load gcc/6.1

python FF_lalazar_model.py WGAN True 100 2 UNIFO