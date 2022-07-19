#!/usr/local_rwth/bin/zsh
 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=CI-Batchjob
#SBATCH --output=CI-Batchjob.%J.out
####SBATCH --account=rwth0792
#SBATCH --time=00:05:00
#SBATCH --partition=c18m
 
 
echo "Hello World!"
