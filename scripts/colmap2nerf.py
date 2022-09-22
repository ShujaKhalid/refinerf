#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--video", default="", help="input path to the video")
    parser.add_argument("--images", default="",
                        help="input path to the images folder, ignored if --video is provided")
    parser.add_argument("--run_colmap", action="store_true",
                        help="run colmap first on the image folder")

    parser.add_argument("--dynamic", action="store_true",
                        help="for dynamic scene, extraly save time calculated from frame index.")
    parser.add_argument("--estimate_affine_shape", action="store_true",
                        help="colmap SiftExtraction option, may yield better results, yet can only be run on CPU.")
    parser.add_argument('--hold', type=int, default=8,
                        help="hold out for validation every $ images")

    parser.add_argument("--video_fps", default=5)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")

    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive", "sequential", "spatial", "transitive",
                        "vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--skip_early", default=0,
                        help="skip this many images from the start")

    parser.add_argument("--colmap_text", default="colmap_text",
                        help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--colmap_db", default="colmap.db",
                        help="colmap database filename")
    parser.add_argument("--base", default=".",
                        help="colmap database filename")
    parser.add_argument("--dataset", default="nvidia",
                        help="dataset")
    parser.add_argument("--mode", default="val",
                        help="mode")
    parser.add_argument("--W", type=int, default=480,
                        help="Rescale width")
    parser.add_argument("--H", type=int, default=270,
                        help="Rescale height")
    args = parser.parse_args()
    return args


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def run_ffmpeg(args):
    video = args.video
    images = args.images
    fps = float(args.video_fps) or 1.0
    args.MULTI_IMG_TRN = False

    print(
        f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    # if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
    #     sys.exit(1)

    try:
        shutil.rmtree(images)
    except:
        pass

    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"

    do_system(
        f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}, scale={str(args.W)}:{str(args.H)}\" {images}/%05d.jpg")


def run_ffmpeg_images(args):

    # TODO: remove hard-coded paths once we have a proof of concept
    base = args.images.split("images")[0]
    max_imgs = 120
    MULTI_IMG_TRN = False
    args.MULTI_IMG_TRN = MULTI_IMG_TRN
    LARGE_DATASET_TRN = True  # 24 images to train

    if (args.mode == "train"):
        new_loc = base + "images_scaled"
        new_loc_mask = base + "motion_masks"
        prod = base.split("/")[-3]
        if (LARGE_DATASET_TRN):
            query_loc = "/home/skhalid/Documents/datalake/dynamic_scene_data_full_bkp/nvidia_data_full/" + \
                prod+"/dense/mv_images"
            query_loc_masks = "/home/skhalid/Documents/datalake/dynamic_scene_data_full_bkp/nvidia_data_full/" + \
                prod+"/dense/mv_masks"
            print("query_loc: {}".format(query_loc))
            os.system("mkdir -p "+new_loc)
            os.system("mkdir -p "+new_loc_mask)
            all_imgs = [glob.glob(query_loc+"/"+str(v).zfill(5)+"/cam"+str((v//2)+1).zfill(2)+".jpg")[0]
                        if (v % 2 == 0)
                        else glob.glob(query_loc+"/"+str(v).zfill(5)+"/cam"+str((v//2)+1).zfill(2)+".jpg")[0]
                        for v in range(24)]
            all_masks = [glob.glob(query_loc_masks+"/"+str(v).zfill(5)+"/cam"+str((v//2)+1).zfill(2)+".png")[0]
                         if (v % 2 == 0)
                         else glob.glob(query_loc_masks+"/"+str(v).zfill(5)+"/cam"+str((v//2)+1).zfill(2)+".png")[0]
                         for v in range(24)]
            print(all_imgs)
            # all_imgs = [v for v in glob.glob(query_loc+"/*")]
        else:
            val_base = "/home/skhalid/Documents/datalake/data/" + prod
            query_loc = val_base + "/images_2"
            query_loc_mask = val_base + "/motion_masks"
            print("query_loc: {}".format(query_loc))
            os.system("mkdir -p "+new_loc)
            os.system("mkdir -p "+new_loc_mask)
            all_imgs = glob.glob(query_loc+"/*.png")
            all_masks = glob.glob(query_loc_mask+"/*.png")
        print(all_imgs)
        print(all_masks)
        all_imgs.sort()
        all_masks.sort()
        all_imgs_qty = len(all_imgs)

        if (MULTI_IMG_TRN):
            for index in range(max_imgs):
                fn = all_imgs[index//(all_imgs_qty+1)]
                out = new_loc+"/"+"00"+str(index).zfill(3)+".jpg"
                cmd = "ffmpeg -i "+fn+" -vf scale=" + \
                    str(args.W)+":"+str(args.H) + " " + out
                print("cmd: {}".format(cmd))
                os.system(cmd)
        else:
            if (LARGE_DATASET_TRN):
                for k, img in enumerate(all_imgs):
                    fn = img.split("/")[-2]
                    fn = fn + ".jpg"
                    out = new_loc+"/"+fn  # Original count

                    cmd = "ffmpeg -i " + img + " -vf scale=" + \
                        str(args.W)+":"+str(args.H) + " " + base+"images/"+fn
                    os.system(cmd)

                    cmd = "ffmpeg -i "+img+" -vf scale=" + \
                        str(args.W)+":"+str(args.H) + " " + out
                    print(cmd)
                    os.system(cmd)
                for k, mask in enumerate(all_masks):
                    fn = mask.split("/")[-2]
                    fn = fn + ".png"
                    out = new_loc+"/"+fn  # Original count
                    cmd = "ffmpeg -i "+mask+" -vf scale=" + \
                        str(args.W)+":"+str(args.H) + " " + out
                    print(cmd)
                    os.system(cmd)
            else:
                # Copy over all of the images
                for img in all_imgs:
                    new_img = "00" + img.split("/")[-1].split(".")[0] + ".jpg"
                    #cmd = "cp -pr " + img + " " + base+"images"
                    cmd = "ffmpeg -i " + img + " -vf scale=" + \
                        str(1280)+":"+str(720) + " " + base+"images/"+new_img
                    os.system(cmd)

                for k, img in enumerate(all_imgs):
                    fn = img.split("/")[-1]
                    out = new_loc+"/"+"00" + \
                        fn.split(".")[0]+".jpg"  # Original count
                    cmd = "ffmpeg -i "+img+" -vf scale=" + \
                        str(args.W)+":"+str(args.H) + " " + out
                    print(cmd)
                    os.system(cmd)
                for k, img in enumerate(all_masks):
                    fn = img.split("/")[-1]
                    out = new_loc_mask+"/"+"00" + \
                        fn.split(".")[0]+".png"  # Original count
                    cmd = "ffmpeg -i "+img+" -vf scale=" + \
                        str(args.W)+":"+str(args.H) + " " + out
                    print(cmd)
                    os.system(cmd)
        # TODO: validate
        #args.images = new_loc
    else:
        if (args.dataset == "nvidia"):
            new_loc = base + "/" + args.mode + "/"
            prod = base.split("/")[-2]
            trn_base = "/home/skhalid/Documents/torch-ngp/results/gt/" + prod + "/"
            query_loc = trn_base
            os.system("mkdir -p "+new_loc)
            files = glob.glob(query_loc+"/*.png")
            files.sort()
            for indx, file in enumerate(files):
                # fn = "v000t0"+str(indx).zfill(2)+".jpg"
                fn = "000"+str(indx).zfill(2)+".jpg"
                # fn = file.split("/")[-1]
                cmd = "ffmpeg -i "+file+" -vf scale=" + \
                    str(args.W)+":"+str(args.H) + " " + new_loc+fn
                #print("cmd: {}".format(cmd))
                os.system(cmd)

            # Use existing validation masks
            query_loc = "/home/skhalid/Documents/datalake/dynamic_scene_data_full_bkp/nvidia_data_full/" + \
                prod+"/dense/"
            query_loc_val = "/home/skhalid/Documents/datalake/dynamic_scene_data_full/nvidia_data_full/" + \
                prod+"/dense/"
            query_loc_imgs = query_loc + "mv_images"
            query_loc_masks = query_loc + "mv_masks"
            query_loc_val_masks = query_loc_val + "motion_masks_val/"

            # gt
            # COMMENT TO KEEP OLD GT
            folders = glob.glob(query_loc_imgs+"/*")
            print("prod: {}".format(prod))
            print("query_loc: {}".format(query_loc))
            print("folders: {}".format(folders))
            folders.sort()
            for indx, folder in enumerate(folders[:24]):
                pose = 0
                files = glob.glob(folder+"/*.jpg")
                files.sort()
                file = files[pose]
                # fn = "v000t0"+str(indx).zfill(2)+".jpg"
                fn = "v0"+str(pose).zfill(2)+"t0"+str(indx).zfill(2)+".png"
                # fn = file.split("/")[-1]
                cmd = "ffmpeg -i "+file+" -vf scale=" + \
                    str(args.W)+":"+str(args.H) + " " + trn_base + fn
                print("cmd: {}".format(cmd))
                os.system(cmd)

            # masks
            folders_masks = glob.glob(query_loc_masks+"/*")
            folders_masks.sort()
            os.system("mkdir -p "+query_loc_val_masks)

            for indx, folder in enumerate(folders_masks[:24]):
                files = glob.glob(folder+"/*.png")
                files.sort()
                file = files[0]
                # fn = "v000t0"+str(indx).zfill(2)+".jpg"
                fn = "000"+str(indx).zfill(2)+".png"
                # fn = file.split("/")[-1]
                cmd = "ffmpeg -i "+file+" -vf scale=" + \
                    str(args.W)+":"+str(args.H) + " " + query_loc_val_masks+fn
                print("cmd: {}".format(cmd))
                os.system(cmd)

            args.images = new_loc
        elif (args.dataset == "custom"):
            new_loc = base + args.mode + "/"
            query_loc = "/home/skhalid/Documents/datalake/dnerf/" + args.dataset + "/images/"
            os.system("mkdir -p "+new_loc)
            files = glob.glob(query_loc+"/*.jpg")
            files.sort()
            for indx, file in enumerate(files):
                # fn = "v000t0"+str(indx).zfill(2)+".jpg"
                fn = "000"+str(indx).zfill(2)+".jpg"
                # fn = file.split("/")[-1]
                cmd = "ffmpeg -i "+file+" -vf scale=" + \
                    str(960)+":"+str(540) + " " + new_loc+fn
                #print("cmd: {}".format(cmd))
                os.system(cmd)


def run_colmap(args):
    db = args.colmap_db
    images = args.images
    text = args.colmap_text
    flag_EAS = int(args.estimate_affine_shape)  # 0 / 1

    db_noext = str(Path(db).with_suffix(""))
    sparse = db_noext + "_sparse"

    print(
        f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    # if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
    #     sys.exit(1)
    if os.path.exists(db):
        os.remove(db)

    do_system(
        # f"colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape {flag_EAS} --SiftExtraction.domain_size_pooling {flag_EAS} --ImageReader.single_camera 1 --SiftExtraction.max_num_features 100000 --database_path {db} --image_path {images}")
        f"colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape {flag_EAS} --SiftExtraction.domain_size_pooling {flag_EAS} --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
    do_system(
        f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching {flag_EAS} --SiftMatching.confidence 0.1 --database_path {db}")
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    do_system(
        f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    do_system(
        f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(
        f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])


def rotmat(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom


if __name__ == "__main__":
    args = parse_args()

    if args.video != "" and args.mode == "train":
        root_dir = os.path.dirname(args.video)
        args.images = os.path.join(root_dir, "images")  # override args.images
        run_ffmpeg(args)
    elif (args.mode == "val"):
        print(args.images)
        args.images = args.images[:-
                                  1] if args.images[-1] == '/' else args.images
        root_dir = os.path.dirname(args.images)
        print(root_dir)

        run_ffmpeg_images(args)
    else:
        # remove trailing / (./a/b/ --> ./a/b)
        args.images = args.images[:-
                                  1] if args.images[-1] == '/' else args.images
        root_dir = os.path.dirname(args.images)
        print(root_dir)
        print(args.images)
        run_ffmpeg_images(args)

    if (args.mode == "val"):
        root_dir = root_dir + "/dense/"

    args.colmap_db = os.path.join(root_dir, args.colmap_db)
    args.colmap_text = os.path.join(root_dir, args.colmap_text)

    if args.run_colmap and args.mode != "val":
        # if args.run_colmap:
        print("RUNNING COLMAP!!!")
        run_colmap(args)

    SKIP_EARLY = int(args.skip_early)
    TEXT_FOLDER = args.colmap_text

    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(
        f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0

        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

        frames = []

        up = np.zeros(3)

        MAX_VAL = 12  # (FOR NVIDIA only)

        for line in f:
            line = line.strip()

            if line[0] == "#":
                continue

            i = i + 1

            # The NVIDIA assertion
            if i < SKIP_EARLY*2:
                continue

            if i % 2 == 1:
                # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                elems = line.split(" ")

                name = '_'.join(elems[9:])
                full_name = os.path.join(args.images, name)
                rel_name = full_name[len(root_dir):]

                # if (args.MULTI_IMG_TRN):
                #     # sk_debug <======================================
                #     img_qty = 12+1
                #     dm = int(full_name.split("/")[-1].split(".")[0])
                #     if (dm % img_qty != 0):
                #         continue
                #     factor = dm//img_qty
                #     full_name = full_name.split(
                #         "/00")[0]+"/"+str(factor).zfill(5)+".jpg"
                #     rel_name = full_name[len(root_dir) + 1:]
                #     print("\n\n\ndm: {}".format(dm))
                #     print("full_name: {}\n\n\n".format(full_name))
                #     # sk_debug <======================================

                print(args.images)
                if (not os.path.isfile(full_name)):
                    continue

                b = sharpness(full_name)
                # print(name, "sharpness =",b)

                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)

                c2w[0:3, 2] *= -1  # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
                c2w[2, :] *= -1  # flip whole world upside down

                up += c2w[0:3, 1]

                frame = {
                    "file_path": rel_name,
                    "sharpness": b,
                    "transform_matrix": c2w
                }

                # print(frame)

                frames.append(frame)

    N = len(frames)
    up = up / np.linalg.norm(up)

    print("[INFO] up vector was", up)

    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in frames:
        f["transform_matrix"] = np.matmul(
            R, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
    print("[INFO] computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in frames:
        mf = f["transform_matrix"][0:3, :]
        for g in frames:
            mg = g["transform_matrix"][0:3, :]
            p, weight = closest_point_2_lines(
                mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            # print(weight)
            if weight > 0.001:  # TODO: used to be 0.01
                totp += p * weight
                totw += weight
    totp /= totw
    for f in frames:
        f["transform_matrix"][0:3, 3] -= totp
    avglen = 0.
    for f in frames:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= N
    print("[INFO] avg camera distance from origin", avglen)
    for f in frames:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    # sort frames by id
    frames.sort(key=lambda d: d['file_path'])

    # add time if scene is dynamic
    if args.dynamic:
        for i, f in enumerate(frames):
            f['time'] = i / N

    for f in frames:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    # if (args.mode == "train"):
    #     fn = all_imgs[index//(all_imgs_qty+1)]

    # sk_debug
    if (args.mode == "val"):
        # Save first pose for validation
        tm = frames[0]["transform_matrix"]
        print(tm)
        for f in frames:
            f["transform_matrix"] = tm

    # construct frames

    def write_json(filename, frames):

        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": str(args.W),  # TODO - 480/270 or 960/540 or 320/240
            "h": str(args.H),  # TODO - 480/270 or 960/540 or 320/240
            "frames": frames,
        }

        output_path = os.path.join(root_dir, filename)
        # print("frames: {}".format(frames))
        if (args.dataset == "nvidia" or args.dataset == "custom"):
            BASE = args.images.split("images_")[0]
            imgs = [v["file_path"] for v in frames]
            #folder = output_path.split("/")[-1].split(".")[0].split("_")[-1]
            #print("folder: {}".format(folder))
            # for img in imgs:
            #     cmd = "cp -pr " + img + " " + \
            #         BASE+"/"+args.mode+"/"+img.split("/")[-1]
            #     print(cmd)
            #     os.system(cmd)
            print(f"[INFO] writing {len(frames)} frames to {output_path}")
            with open(output_path, "w") as outfile:
                json.dump(out, outfile, indent=2)
        else:
            imgs = [v["file_path"] for v in frames]
            #folder = output_path.split("/")[-1].split(".")[0].split("_")[-1]
            BASE = args.images.split("images_")[0]
            # print("folder: {}".format(folder))
            for img in imgs:
                cmd = "cp -pr " + "/"+img+" " + \
                    BASE+"/"+args.mode+"/"+img.split("/")[-1]
                print(cmd)
                os.system(cmd)
            print(f"[INFO] writing {len(frames)} frames to {output_path}")
            with open(output_path, "w") as outfile:
                json.dump(out, outfile, indent=2)

    # just one transforms.json, don't do data split
    if args.hold <= 0:

        write_json('transforms.json', frames)

    else:
        all_ids = np.arange(N)
        test_ids = all_ids[::args.hold]
        train_ids = np.array([i for i in all_ids if i not in test_ids])
        W = args.W
        H = args.H

        if (args.dataset == "nvidia" or args.dataset == "custom"):
            frames_all = [f for i, f in enumerate(frames) if i in all_ids]
            # frames_all = glob.glob(args.images+"/*.jpg")
            print(frames_all)
            write_json('transforms_'+args.mode+'.json', frames_all)

        else:
            frames_train = [f for i, f in enumerate(frames) if i in train_ids]
            frames_test = [f for i, f in enumerate(frames) if i in test_ids]

            write_json('transforms_train.json', frames_train)
            write_json('transforms_val.json', frames_test[::10])
            write_json('transforms_test.json', frames_test)
