# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
sys.path.insert(0, './')

import argparse
import cv2, os

from fcos_core.config import cfg
from demo.predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/embed_mask/embed_mask_R50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="models/embed_mask_R50_1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--out-dir",
        default="demo/output",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [
        0.24445024132728577, 0.2556260824203491, 0.2336651235818863, 0.26643890142440796, 0.22829005122184753,
        0.27605465054512024, 0.29680299758911133, 0.24539557099342346, 0.22566702961921692, 0.21125544607639313,
        0.3632965385913849, 0.42116600275039673, 0.29700127243995667, 0.2278410643339157, 0.2317150980234146,
        0.30244436860084534, 0.32276564836502075, 0.25707629323005676, 0.24852260947227478, 0.24491029977798462,
        0.2518414556980133, 0.35320255160331726, 0.2866332232952118, 0.2207552194595337, 0.2568267285823822,
        0.24461865425109863, 0.20570527017116547, 0.2656995356082916, 0.21232444047927856, 0.2799481451511383,
        0.18180416524410248, 0.2654014825820923, 0.262266606092453, 0.19924932718276978, 0.22213412821292877,
        0.3075449764728546, 0.2290934920310974, 0.2963321805000305, 0.23535756766796112, 0.2430417388677597,
        0.22808006405830383, 0.2716907560825348, 0.21096138656139374, 0.18565504252910614, 0.17213594913482666,
        0.2755044996738434, 0.22538238763809204, 0.22792285680770874, 0.24877801537513733, 0.23092558979988098,
        0.23993775248527527, 0.21917308866977692, 0.2535002529621124, 0.30203622579574585, 0.19476301968097687,
        0.24782243371009827, 0.22699865698814392, 0.25022363662719727, 0.23006463050842285, 0.22317998111248016,
        0.20648975670337677, 0.28253015875816345, 0.35304051637649536, 0.2882220447063446, 0.2875506281852722,
        0.21613512933254242, 0.308322936296463, 0.29409125447273254, 0.3021804690361023, 0.273112416267395,
        0.23458659648895264, 0.2998719811439514, 0.2715963125228882, 0.1898047924041748, 0.32565683126449585,
        0.25560101866722107, 0.265905499458313, 0.3087238669395447, 0.2053961306810379, 0.20331673324108124
    ]

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imwrite(os.path.join(args.out_dir, im_name), composite)
    print("Press any keys to exit ...")

if __name__ == "__main__":
    main()

