#!/bin/bash

rm logfiles/*

qsub -q generic ./runfiles/runfile_2.sh
qsub -q generic ./runfiles/runfile_3.sh
qsub -q generic ./runfiles/runfile_5.sh
qsub -q generic ./runfiles/runfile_6.sh
