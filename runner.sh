#!/bin/bash

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda

DATASET_PATH="../datalake/dnerf/custom"
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
		echo "Running custom module..."
	    # We're dealing with a custom video
		DATASET_PATH="../datalake/dnerf/custom"
		FILENAME=$(basename "$2" .mp4)

		mkdir -p $DATASET_PATH
		mkdir -p $DATASET_PATH/train
		mkdir -p $DATASET_PATH/val
		mkdir -p $DATASET_PATH/test
		mkdir -p $DATASET_PATH/images_colmap
		
		#python ./utils/generate_data.py --videopath $2 --data_dir $DATASET_PATH
		python scripts/colmap2nerf.py --video "$2" --run_colmap --dynamic

		echo $FILENAME
		for i in $DATASET_PATH/images/*.png ; do convert "$i" "${i%.*}.jpg" ; done
		# cp -pr $DATASET_PATH/images/*.jpg $DATASET_PATH/train
		cp -pr $DATASET_PATH/images/*.jpg $DATASET_PATH/images_colmap
	else
		mkdir -p $DATASET_PATH/images_colmap
		for i in $DATASET_PATH/train/*.png ; do convert "$i" "${i%.*}.jpg" ; done
		cp -pr $DATASET_PATH/train/*.jpg $DATASET_PATH/images_colmap

		colmap feature_extractor \
		--database_path $DATASET_PATH/database.db \
		--image_path $DATASET_PATH/images_colmap \
		--ImageReader.mask_path $DATASET_PATH/background_mask \
		--ImageReader.camera_model "SIMPLE_RADIAL" \
		--SiftExtraction.max_num_features 100000
		# --ImageReader.single_camera 1

		colmap exhaustive_matcher \
		--database_path $DATASET_PATH/database.db \
		--SiftMatching.confidence 0.01
		# --SiftMatching.max_num_matches 100

		mkdir $DATASET_PATH/sparse
		colmap mapper \
		--database_path $DATASET_PATH/database.db \
		--image_path $DATASET_PATH/images_colmap \
		--output_path $DATASET_PATH/sparse \
		--Mapper.num_threads 16 \
		--Mapper.init_min_tri_angle 6 \
		--Mapper.multiple_models 0 \
		--Mapper.extract_colors 0
	fi


	python utils/generate_depth.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_MIDAS
	python utils/generate_flow.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_RAFT 
	python utils/generate_motion_mask.py --dataset_path $DATASET_PATH
fi

if [[ "$1" == "--run" || "$2" == "--run" || "$3" == "--run"  ]]
then
	python main_dnerf.py $DATASET_PATH --workspace trial_nerf --fp16 --cuda_ray $GUIFLAG
fi
