#!/bin/bash

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda

DATASET_PATH="../datalake/dnerf/jumpingjacks"
NM_WEIGHTS="/home/skhalid/Documents/datalake/neural_motion_weights/"
WEIGHTS_MIDAS=$NM_WEIGHTS"midas_v21-f6b98070.pt"
WEIGHTS_RAFT=$NM_WEIGHTS"raft-things.pth"
#cd raymarching 
#pip install .
#cd ..

if [[ "$1" == "--gui" || "$2" == "--gui" || "$3" == "--gui" ]]
then
	GUIFLAG="--gui"
fi

if [[ "$1" == "--extract" ]]
then
	if [[ -f "$2" ]]
	then
		python ./utils/generate_data.py --videopath $2
	else
		mkdir -p $DATASET_PATH/images_colmap
		cp -pr $DATASET_PATH/train/*.png $DATASET_PATH/images_colmap
	fi

	# colmap feature_extractor \
	# --database_path $DATASET_PATH/database.db \
	# --image_path $DATASET_PATH/images_colmap \
	# --ImageReader.mask_path $DATASET_PATH/background_mask \
	# --ImageReader.single_camera 1

	# colmap exhaustive_matcher \
	# --database_path $DATASET_PATH/database.db

	# mkdir $DATASET_PATH/sparse
	# colmap mapper \
	# --database_path $DATASET_PATH/database.db \
	# --image_path $DATASET_PATH/images_colmap \
	# --output_path $DATASET_PATH/sparse \
	# --Mapper.num_threads 16 \
	# --Mapper.init_min_tri_angle 6 \
	# --Mapper.multiple_models 0 \
	# --Mapper.extract_colors 0

	# python ./utils/generate_depth.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_MIDAS
	# python ./utils/generate_flow.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_RAFT 
	python ./utils/generate_motion_mask.py --dataset_path $DATASET_PATH
fi

if [[ "$1" == "--run" || "$2" == "--run" || "$3" == "--run"  ]]
then
	python main_dnerf.py $DATASET_PATH --workspace trial_nerf --fp16 --cuda_ray $GUIFLAG
fi
