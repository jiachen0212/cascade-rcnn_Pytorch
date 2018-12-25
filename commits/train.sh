#/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
logdir=cascade
mkdir -p $logdir

partition=SenseMediaF    # SenseMediaA
num_gpus=8
name=fpn

srun -p ${partition} --mpi=pmi2 --gres=gpu:${num_gpus} -n1 --ntasks-per-node=${num_gpus} --job-name=cascade --kill-on-bad-exit=1 \
python trainval_net.py 2>&1 | tee $logdir/$name-train-$partition-$now.log

