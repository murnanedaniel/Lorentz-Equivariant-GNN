#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu 
#SBATCH -t 1:00:00 
#SBATCH -G 2
#SBATCH -n 2
#SBATCH -c 20
#SBATCH --exclusive
#SBATCH -q special
#SBATCH -o logs/%x-%j.out
#SBATCH -J classification-sweep


conda activate exatrkx-test
export SLURM_CPU_BIND="cores"
echo -e "\nStarting sweeps\n"

# for i in {0..3}; do
#     echo "Launching task $i"
#     srun --gres=craynetwork:0 -n 1 -G 1 wandb agent murnanedaniel/GeneralNet/nkkj4lwv &
# done

# srun -G 1 python do_nothing.py

# srun --gres=craynetwork:0 -n 1 --cpus-per-gpu=10 --gpus-per-task=1 nvidia-smi &
# srun --gres=craynetwork:0 -n 1 --cpus-per-gpu=10 --gpus-per-task=1 nvidia-smi &

srun -n 1 -G 2 nvidia-smi

srun --gres=craynetwork:0 --exact -u -n 1 --gpus-per-task 1 nvidia-smi &
srun --gres=craynetwork:0 --exact -u -n 1 --gpus-per-task 1 nvidia-smi &

# srun --exact -u -n 1 --gpus-per-task 1 -c 1 python do_nothing.py &
# srun --exact -u -n 1 --gpus-per-task 1 -c 1 python do_nothing.py &
wait

# parallel -j2 'CUDA_VISIBLE_DEVICES=$(("{%}" - 1)) && wandb agent murnanedaniel/GeneralNet/nkkj4lwv'
