#!/bin/bash

#SBATCH --job-name=sshd
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j_%N.log
#SBATCH --account=free
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
##SBATCH --exclude=ins094
##SBATCH --constraint=h100
##SBATCH --mem=50gb

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "********************************************************************"
echo "Starting sshd in Slurm as user"
echo "Environment information:"
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Node IP:" $(hostname -i)
echo "Path:" $(pwd)
echo "Listening on:" $PORT
echo "********************************************************************"

/usr/sbin/sshd -D -p ${PORT} -f /dev/null -h ${HOME}/.ssh/id_ed25519
