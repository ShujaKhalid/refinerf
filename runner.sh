#!/bin/bash

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda

DATASET_PATH="../datalake/dnerf/jumpingjacks/"

#cd raymarching 
#pip install .
#cd ..

if [[ "$1" == "--gui" || "$2" == "--gui" || "$3" == "--gui" ]]
then
	GUIFLAG="--gui"
fi

if [[ "$1" == "--extract" ]]
then
	python ./utils/generate_depth.py --dataset_path $DATASET_PATH --model weights/midas_v21-f6b98070.pt
	python ./utils/generate_flow.py --dataset_path $DATASET_PATH --model weights/raft-things.pth
	python ./utils/generate_motion_mask.py --dataset_path $DATASET_PATH
fi

if [[ "$1" == "--run" || "$2" == "--run" || "$3" == "--run"  ]]
then
	python main_dnerf.py $DATASET_PATH  --workspace trial_nerf --fp16 --cuda_ray $GUIFLAG
fi
