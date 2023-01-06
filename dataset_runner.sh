#!/bin/bash

cases=('Playground' 'Balloon1' 'Balloon2' 'Umbrella' 'Truck' 'Jumping')

for item in "${cases[@]}";
do
	#echo $item
	rm -rf $item/checkpoints/* && ./runner.sh --run $item
done

