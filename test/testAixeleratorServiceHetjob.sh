#!/usr/bin/zsh

#SBATCH --account=rwth0792
#SBATCH --time=00:05:00
#SBATCH --output=aixeleratorService-hetjob-gpu.%J.out
#SBATCH --exclusive
#SBATCH --partition=c18g
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks=2
#SBATCH hetjob
#SBATCH --output=aixeleratorService-hetjob-cpu.%J.out
#SBATCH --exclusive
#SBATCH --partition=c18m
#SBATCH --ntasks=2

module switch intel gcc/8
module load intel
module load cuda/11.4
module load cudnn/8.3.2


cd /home/rwth0792/aixeleratorservice
cd BUILD

AIX_EXE=/home/rwth0792/aixeleratorservice/BUILD/test/testAIxeleratorService.cpp.x

srun ${AIX_EXE} : ./test/module-wrapper.sh ${AIX_EXE}
