GPUS=$1
python3 -m torch.distributed.launch --master_port=12346 --nproc_per_node=$GPUS main.py ${@:2}