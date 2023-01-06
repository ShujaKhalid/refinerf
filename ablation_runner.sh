#!/bin/bash

#cases=('Playground' 'Balloon1' 'Balloon2' 'Umbrella' 'Truck' 'Jumping')
cases=('Umbrella')
deform_dims=(3 5 7 9)
time_dims=(0 1 3)
deform_qty_arr=(4 8)
deform_hidden_qty_arr=(64 128 256)
iters=10000

for scene in "${cases[@]}";
do
	DATASET_PATH="/home/skhalid/Documents/datalake/dynamic_scene_data_full/nvidia_data_full/$scene/dense"
	for time_dim in "${time_dims[@]}";
	do
		for deform_dim in "${deform_dims[@]}";
		do
			for deform_qty in "${deform_qty_arr[@]}";
			do
				for deform_hidden_qty in "${deform_hidden_qty_arr[@]}";
				do
					rm -rf $scene/checkpoints/*
					tensorboard_folder=$scene"_encoder_deform_"$deform_dim"_time_dim_"$time_dim"_deform_qty_"$deform_qty"_deform_hidden_qty_"$deform_hidden_qty"_iters_"$iters
					python main_dnerf.py $DATASET_PATH \
						--workspace $scene \
						--tensorboard_folder $tensorboard_folder \
						--encoder_s_fact 10 \
						--encoder_dir_s_fact 4  \
						--encoder_d_fact 10 \
						--encoder_dir_d_fact 4 \
						--encoder_d_constant 1  \
						--encoder_deform $deform_dim  \
						--encoder_time $time_dim \
						--num_layers 2 \
						--hidden_dim 256 \
						--geo_feat_dim 64 \
						--num_layers_color 3 \
						--hidden_dim_color 256  \
						--num_layers_deform $deform_qty \
						--hidden_dim_deform $deform_hidden_qty \
						--iters $iters \
						--fp16 \
						-O
					cp -pr /home/skhalid/Documents/torch-ngp/results/Ours/$scene/* ./$scene/run/ngp/$tensorboard_folder
				done
			done
		done
	done
done

