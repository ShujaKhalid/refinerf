#!/bin/bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda

# pip install -r requirements.txt
# pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# cd raymarching 
# pip install .
# cd ..

# cd gridencoder
# pip install .
# cd ..

# cd shencoder
# pip install .
# cd .. 

# cd ffmlp
# pip install .
# cd ..

# cd detectron2 
# pip install .
# cd ..

#cases=('Playground' 'Balloon1' 'Balloon2' 'Umbrella' 'Truck' 'Jumping')
#cases=('Umbrella' 'Playground' 'Balloon1' 'Balloon2' 'Skating' 'Jumping')
cases=('Umbrella' 'Playground')
#cases=('Playground')
deform_dims=(9)
time_dims=(3)
deform_qty_arr=(4)
deform_hidden_qty_arr=(512)
deform_intrinsics_arr=(0)
# noise_pct_arr=(1.0)
deform_extrinsics_arr=(0)
noise_pct_arr=(0.0)
barf_arr=(0)
nerfmm_arr=(0)
# noise_pct_arr=(0.1 0.05 0.025 0.0)
# noise_pct_arr=(0)
iters=12000

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
					for intrinsics in "${deform_intrinsics_arr[@]}";
					do
						for extrinsics in "${deform_extrinsics_arr[@]}";
						do
							for noise_pct in "${noise_pct_arr[@]}";
							do
								for barf in "${barf_arr[@]}";
								do
									for nerfmm in "${nerfmm_arr[@]}";
									do
										#out_folder="refinerf" 
										out_folder=$scene
										rm -rf $out_folder/checkpoints/*
										mkdir -p $out_folder
										tensorboard_folder=$scene"_encoder_deform_"$deform_dim"_time_dim_"$time_dim"_deform_qty_"$deform_qty"_deform_hidden_qty_"$deform_hidden_qty"_iters_"$iters"_intrinsics_"$intrinsics"_extrinsics_"$extrinsics"_noise_pct_"$noise_pct"_barf_"$barf"_nerfmm_"$nerfmm
										python main_dnerf.py $DATASET_PATH \
											--workspace $out_folder \
											--tensorboard_folder $tensorboard_folder \
											--encoder_s_fact 10 \
											--encoder_dir_s_fact 10  \
											--encoder_d_fact 10 \
											--encoder_dir_d_fact 10 \
											--encoder_d_constant 1  \
											--encoder_time $time_dim \
											--encoder_deform $deform_dim  \
											--num_layers 2 \
											--hidden_dim 256 \
											--geo_feat_dim 64 \
											--num_layers_color 3 \
											--hidden_dim_color 256  \
											--num_layers_deform $deform_qty \
											--hidden_dim_deform $deform_hidden_qty \
											--iters $iters \
											--noise_pct $noise_pct \
											--fp16 \
											--pred_intrinsics $intrinsics \
											--pred_extrinsics $extrinsics \
											--barf $barf \
											--nerfmm $nerfmm \
											-O
										cp -pr /home/skhalid/Documents/torch-ngp/results/Ours/$out_folder/* ./$out_folder/run/ngp/$tensorboard_folder
									done
								done										
							done
						done
					done
				done
			done
		done
	done
done

