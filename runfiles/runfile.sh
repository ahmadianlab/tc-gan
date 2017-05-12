#!/bin/bash

qsub -q generic -v loss="CE",layer="[]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="CE",layer="[128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="CE",layer="[128,128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="CE",layer="[128,128,128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="LS",layer="[]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="LS",layer="[128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="LS",layer="[128,128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
qsub -q generic -v loss="LS",layer="[128,128,128]",nsam="20",io="asym_linear",rcost="100." ./runfiles/runfile_arg.sh
