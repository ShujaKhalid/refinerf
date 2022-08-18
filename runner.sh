#!/bin/bash

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda

# DATASET_PATH="../datalake/dnerf/bouncingballs"
# SCENE="bouncingballs"

# DATASET_PATH="../datalake/dnerf/custom"
# SCENE="DynamicFace-2"
# SCENE="Truck-2"
SCENE="Umbrella"
# SCENE="Jumping"
# SCENE="Balloon2"
# SCENE="Balloon2-2"
# SCENE="Skating-2"
# SCENE="Playground"
DATASET_PATH="/home/skhalid/Documents/datalake/dynamic_scene_data_full/nvidia_data_full/$SCENE/dense"

NM_WEIGHTS="/home/skhalid/Documents/datalake/neural_motion_weights/"
WEIGHTS_MIDAS=$NM_WEIGHTS"midas_v21-f6b98070.pt"
WEIGHTS_RAFT=$NM_WEIGHTS"raft-things.pth"

# Run when making changes to the CUDA kernels
#cd raymarching 
#pip install .
#cd ..

if [[ "$1" == "--gui" || "$2" == "--gui" || "$3" == "--gui" ]]
then
	GUIFLAG="--gui"
fi

if [[ "$1" == "--extract" ]]
then

	mkdir -p $DATASET_PATH
	mkdir -p $DATASET_PATH/train
	mkdir -p $DATASET_PATH/val
	mkdir -p $DATASET_PATH/test
	mkdir -p $DATASET_PATH/images_colmap

	if [[ -f "$2" ]]
	then
		echo "Running custom module..."
	    # We're dealing with a custom video
		DATASET_PATH="../datalake/dnerf/custom"
		FILENAME=$(basename "$2" .mp4)

		#python ./utils/generate_data.py --videopath $2 --data_dir $DATASET_PATH
		python scripts/colmap2nerf.py --video "$2" --run_colmap --dynamic

		echo $FILENAME
		# for i in $DATASET_PATH/images/*.png ; do convert "$i" "${i%.*}.jpg" ; done
		# cp -pr $DATASET_PATH/images/*.jpg $DATASET_PATH/train
		cp -pr $DATASET_PATH/images/*.jpg $DATASET_PATH/images_colmap
	else
		if [[ "$2" == "--nvidia" ]]
		then
			echo "\\n\\n COLMAP2NERF \\n\\n"
			mv $DATASET_PATH/sparse /tmp
			rm -rf $DATASET_PATH/*
			mv /tmp/sparse $DATASET_PATH/
			cp -pr $DATASET_PATH/sparse $DATASET_PATH/colmap_sparse
			mkdir -p $DATASET_PATH/images
			mkdir -p $DATASET_PATH/images_colmap
			IMAGE_PTH="images"
			python scripts/colmap2nerf.py --images $DATASET_PATH/$IMAGE_PTH --run_colmap --dynamic --dataset nvidia --mode train
			python scripts/colmap2nerf.py --images $DATASET_PATH/$IMAGE_PTH --run_colmap --dynamic --dataset nvidia --mode val
			cp -pr $DATASET_PATH/images_scaled/*.jpg $DATASET_PATH/images_colmap
		else
			mkdir -p $DATASET_PATH/images_colmap
			for i in $DATASET_PATH/train/*.png ; do convert "$i" "${i%.*}.jpg" ; done
			cp -pr $DATASET_PATH/train/*.jpg $DATASET_PATH/images_colmap

			# python scripts/colmap2nerf.py --images $DATASET_PATH/images_colmap --run_colmap --dynamic

			colmap feature_extractor \
			--database_path $DATASET_PATH/database.db \
			--image_path $DATASET_PATH/images_colmap \
			--ImageReader.mask_path $DATASET_PATH/background_mask \
			--ImageReader.camera_model "SIMPLE_PINHOLE" \
			--SiftExtraction.max_num_features 100000
			# --ImageReader.single_camera 1

			colmap exhaustive_matcher \
			--database_path $DATASET_PATH/database.db \
			--SiftMatching.confidence 0.01
			# --SiftMatching.max_num_matches 100

			mkdir $DATASET_PATH/colmap_sparse
			colmap mapper \
			--database_path $DATASET_PATH/database.db \
			--image_path $DATASET_PATH/images_colmap \
			--output_path $DATASET_PATH/colmap_sparse \
			--Mapper.num_threads 16 \
			--Mapper.init_min_tri_angle 6 \
			--Mapper.multiple_models 0 \
			--Mapper.extract_colors 0
		fi
	fi

	# # train
	python utils/generate_depth.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_MIDAS --input_folder images_colmap --output_folder disp --output_img_folder disp_img 
	# python utils/generate_flow.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_RAFT --input_folder images_colmap --output_folder flow --output_img_folder flow_img 
	# python utils/generate_motion_mask.py --dataset_path $DATASET_PATH --input_folder images_colmap --output_sem_mask_folder semantic_mask --output_mot_seg_folder motion_segmentation --output_mot_mask_folder motion_masks

	# # val
	python utils/generate_depth.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_MIDAS --input_folder val --output_folder disp_val --output_img_folder disp_img_val 
	# python utils/generate_flow.py --dataset_path $DATASET_PATH$CASE --model $WEIGHTS_RAFT --input_folder val --output_folder flow_val --output_img_folder flow_img_val 
	# python utils/generate_motion_mask.py --dataset_path $DATASET_PATH --input_folder val --output_sem_mask_folder semantic_mask_val --output_mot_seg_folder motion_segmentation_val --output_mot_mask_folder motion_masks_val

fi

if [[ "$1" == "--run" || "$2" == "--run" || "$3" == "--run"  ]]
then
	python main_dnerf.py $DATASET_PATH --workspace $SCENE --fp16 -O  $GUIFLAG
fi
