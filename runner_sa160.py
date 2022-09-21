import os
import glob
import tqdm
import numpy as np

DATASET = "sa160"
DATASET_PATH = "/home/skhalid/Documents/datalake/sa160/"
RESULTS_FOLDER = "/home/skhalid/Documents/torch-ngp/results/Ours/custom"
BASE = "/home/skhalid/Documents/datalake/dnerf/custom/"
DEST = "/home/skhalid/Desktop/" + DATASET + "/"
all_clips = glob.glob(DATASET_PATH+"/*/*.mp4")


for clip in tqdm.tqdm(all_clips):
    #print("Extracting clip: {}".format(clip))
    case = clip.split("/")[-2] + "_" + clip.split("/")[-1].split(".")[0]
    new_dest = DEST + case

    # ==> Create the new folder and add the results to it
    cmd = "mkdir -p " + new_dest
    os.system(cmd)

    # ==> Copy the clip over with the new name
    cmd = "cp -p " + clip + " " + BASE + "clippy1.mp4"
    os.system(cmd)

    # ==> Run colmap
    # - extraction and resuired files
    cmd = "./runner.sh --extract " + BASE + \
        "clippy1.mp4 clippy1.mp4 --dataset custom"
    os.system(cmd)

    # ==> Run the model
    cmd = "rm -rf custom/checkpoints/* && time ./runner.sh --run"
    os.system(cmd)

    cmd = "mv " + RESULTS_FOLDER + "/*.png " + new_dest + "/"
    os.system(cmd)
    print(cmd)
