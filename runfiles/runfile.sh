#!/bin/bash

qsub -q generic ./runfiles/runfile_1_GAN.sh
qsub -q generic ./runfiles/runfile_2_GAN.sh
qsub -q generic ./runfiles/runfile_1_WGAN.sh
qsub -q generic ./runfiles/runfile_2_WGAN.sh
