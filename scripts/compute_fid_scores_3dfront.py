# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script for computing the FID score between real and synthesized scenes.
"""
import argparse
import os
import sys

import torch

import numpy as np
from PIL import Image

from cleanfid import fid

import shutil

from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_front import CachedThreedFront


class ThreedFrontRenderDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx].image_path
        img = Image.open(image_path)
        return img


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    # parser.add_argument(
    #     "path_to_real_renderings",
    #     help="Path to the folder containing the real renderings"
    # )
    # parser.add_argument(
    #     "path_to_synthesized_renderings",
    #     help="Path to the folder containing the synthesized"
    # )
    # parser.add_argument(
    #     "path_to_annotations",
    #     help="Path to the folder containing the annotations"
    # )
    parser.add_argument(
        "--compare_trainval",
        action="store_true",
        help="if compare trainval"
    )

    args = parser.parse_args(argv)
    room_type = ["bedroom", "livingroom", "diningroom", "all"]
    room = room_type[0]
    print("testing {}...".format(room))
    room_dict = {'bedroom': ["Bedroom", "MasterBedroom", "SecondBedroom"], 'livingroom': ['LivingDiningRoom','LivingRoom'], 'diningroom': ['LivingDiningRoom','DiningRoom']}
    # Create Real datasets
    # config = dict(
    #     train_stats="dataset_stats.txt",
    #     room_layout_size="256,256"
    # )
    # splits_builder = CSVSplitsBuilder(args.path_to_annotations)
    if args.compare_trainval:
        args.path_to_real_renderings = "/media/ymxlzgy/Data/graphto3d_v2_test/gt/small/trainval"
        args.path_to_synthesized_renderings = "/media/ymxlzgy/Data/graphto3d_v2_trainval/g2bv1_small/retrieval/render_imgs"
        # test_real = ThreedFrontRenderDataset(CachedThreedFront(
        #     args.path_to_real_renderings,
        #     config=config,
        #     scene_ids=splits_builder.get_splits(["train", "val"])
        # ))
    else:
        args.path_to_real_renderings = "/media/ymxlzgy/Data/graphto3d_v2_test/sdf_fov90_h8_wo_lamp_no_stool/small/test"
        args.path_to_synthesized_renderings = "/media/ymxlzgy/Data/graphto3d_v2_test/g2sv2_180_crossattn_small_no_stool_backup/render_imgs/v2"
        # test_real = ThreedFrontRenderDataset(CachedThreedFront(
        #     args.path_to_real_renderings,
        #     config=config,
        #     scene_ids=splits_builder.get_splits(["test"])
        # ))

    print("Generating temporary a folder with test_real images...")
    path_to_test_real = "/media/ymxlzgy/Data/graphto3d_v2_test/fid_kid_tmp/real" # /tmp/test_real
    if not os.path.exists(path_to_test_real):
        os.makedirs(path_to_test_real)
    real_images = [
        os.path.join(args.path_to_real_renderings, oi)
        for oi in os.listdir(args.path_to_real_renderings)
        if oi.endswith(".png") and oi.split('-')[0] in room_dict[room]
    ]
    for i, fi in enumerate(real_images):
        shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_real, i))
    # Number of images to be copied
    N = len(real_images)
    print('number of real images :', len(real_images))

    print("Generating temporary a folder with test_fake images...")
    path_to_test_fake = "/media/ymxlzgy/Data/graphto3d_v2_test/fid_kid_tmp/fake" #/tmp/test_fake/
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)

    synthesized_images = [
        os.path.join(args.path_to_synthesized_renderings, oi)
        for oi in os.listdir(args.path_to_synthesized_renderings)
        if oi.endswith(".png") and oi.split('-')[0] in room_dict[room]
    ]
    print('number of synthesized images :', len(synthesized_images))

    scores = []
    scores2 = []
    if args.compare_trainval:
        if True:
            for i, fi in enumerate(synthesized_images):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

            # Compute the FID score
            fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
            print('fid score:', fid_score)
            kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
            print('kid score:', kid_score)
            os.system('rm -r %s'%path_to_test_real)
            os.system('rm -r %s'%path_to_test_fake)
    else:
        for _ in range(1):
            # np.random.shuffle(synthesized_images)
            # synthesized_images_subset = np.random.choice(synthesized_images, N)
            synthesized_images_subset = synthesized_images
            for i, fi in enumerate(synthesized_images_subset):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

            # Compute the FID score
            fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))

            scores.append(fid_score)
            print('iter: {:d}, fid :{:f}'.format(_, fid_score))
            print('iter: {:d}, fid avg: {:f}'.format(_, sum(scores) / len(scores)) )
            print('iter: {:d}, fid std: {:f}'.format(_, np.std(scores)) )

            fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, mode="clean", model_name="clip_vit_b_32")
            print('iter: {:d}, fid-clip :{:f}'.format(_, fid_score))

            kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
            scores2.append(kid_score)
            print('iter: {:d}, kid: {:f}'.format(_, kid_score) )
            print('iter: {:d}, kid avg: {:f}'.format(_, sum(scores2) / len(scores2)) )
            print('iter: {:d}, kid std: {:f}'.format(_, np.std(scores2)) )
        os.system('rm -r %s'%path_to_test_real)
        os.system('rm -r %s'%path_to_test_fake)


if __name__ == "__main__":
    main(None)