#SBATCH -C gpu
#SBATCH -A ntrain4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=01:00:00
#SBATCH --image=nersc/pytorch:ngc-23.07-v0
#SBATCH --module=gpu,nccl-2.18
#SBATCH --reservation=sc23_dl_tutorial_2
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out


# salloc -N 1 --gpus-per-node 4 --ntasks-per-node=4 --cpus-per-task 32 --gpu-bind=none  -t 4:00:00 -q interactive -A nintern -C gpu --image=nersc/pytorch:ngc-23.07-v0 --module=gpu,nccl-2.18
# shifter --image=nersc/pytorch:ngc-23.07-v0 --module=gpu,nccl-2.18 /bin/bash 