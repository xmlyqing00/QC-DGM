#!/bin/bash
#SBATCH -N 1               # request one node
#SBATCH -J QC-DGM-256
#SBATCH -t 3-00:00:00	        # request two hours
#SBATCH -p checkpt          # in single partition (queue)
#SBATCH -A hpc_gvc2021
#SBATCH -o slurm-%j.out-%N # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e slurm-%j.err-%N # optional, name of the stderr, using job and hostname values
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=yqliang@cct.lsu.edu
# below are job commands
date

# Set some handy environment variables.

export HOME_DIR=/home/$USER
export SRC_DIR=QC-DGM

# Make sure the WORK_DIR exists:
#mkdir -p $WORK_DIR
# Copy files, jump to WORK_DIR, and execute a program called "mydemo"
#cp -r $HOME_DIR/$SRC_DIR $WORK_DIR
#cd $WORK_DIR/$SRC_DIR
#rm -rf .git

cd $HOME_DIR/$SRC_DIR

python3 eval.py \
--cfg ./experiments/QCDGM_voc.yaml \
--bs 256

# Mark the time it finishes.
date
# exit the job
exit 0