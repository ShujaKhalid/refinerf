import os
import glob
import tqdm
import numpy as np

DATASET = "sa160"
# DATASET_PATH = "/home/skhalid/Documents/datalake/sa160/"
DATASET_PATH = "/home/skhalid/Desktop/sa160/"
RESULTS_FOLDER = "/home/skhalid/Documents/torch-ngp/results/Ours/custom"
BASE = "/home/skhalid/Documents/datalake/dnerf/custom/"
DEPTH_FOLDER = BASE+"/disp_img_val/"
ORIG_FOLDER = BASE+"/images/"
DEST = "/home/skhalid/Desktop/" + DATASET + "/"
# all_clips = glob.glob(DATASET_PATH+"/*.mp4")
# all_clips = glob.glob(DATASET_PATH+"/*dissection_thermal*/*.mp4") + \
#     glob.glob(DATASET_PATH+"/*cutting*/*.mp4") + \
#     glob.glob(DATASET_PATH+"/*knotting*/*.mp4") + \
#     glob.glob(DATASET_PATH+"/*thread*/*.mp4")
all_clips = glob.glob(DATASET_PATH+"/*.mp4")
MAX_CASES = 1

for clip in tqdm.tqdm(all_clips[:MAX_CASES]):
    # print("Extracting clip: {}".format(clip))
    case = clip.split("/")[-2] + "_" + clip.split("/")[-1].split(".")[0]
    new_dest_orig = DEST + case + "/orig"
    new_dest_recon = DEST + case + "/recon"
    new_dest_depth = DEST + case + "/depth"

    # ==> Create the new folder and add the results to it
    cmd = "mkdir -p " + new_dest_orig
    os.system(cmd)
    cmd = "mkdir -p " + new_dest_recon
    os.system(cmd)
    cmd = "mkdir -p " + new_dest_depth
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

    cmd = "mv " + ORIG_FOLDER + "/*.jpg " + new_dest_orig + "/"
    os.system(cmd)

    cmd = "mv " + RESULTS_FOLDER + "/*.png " + new_dest_recon + "/"
    os.system(cmd)

    cmd = "mv " + DEPTH_FOLDER + "/*.png " + new_dest_depth + "/"
    os.system(cmd)

    print(cmd)
