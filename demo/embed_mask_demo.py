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
        0.24721044301986694, 0.2316334992647171, 0.23782534897327423, 0.2447730302810669, 
        0.26833730936050415, 0.2909756898880005, 0.22202278673648834, 0.23603129386901855, 
        0.19448654353618622, 0.2009030282497406, 0.2205723077058792, 0.4426179826259613, 
        0.2812938094139099, 0.23200270533561707, 0.22222928702831268, 0.34396135807037354, 
        0.29865574836730957, 0.2620207965373993, 0.23538640141487122, 0.21343813836574554, 
        0.23408174514770508, 0.3619556427001953, 0.25181055068969727, 0.2753196656703949, 
        0.20989173650741577, 0.256824254989624, 0.24953776597976685, 0.2482326775789261, 
        0.23516853153705597, 0.3231242001056671, 0.1875445693731308, 0.22903329133987427, 
        0.220603808760643, 0.1938045769929886, 0.2102973908185959, 0.30885136127471924, 
        0.21589471399784088, 0.2611836791038513, 0.27154257893562317, 0.2536311149597168, 
        0.21989859640598297, 0.2741137146949768, 0.24886088073253632, 0.20183633267879486, 
        0.17529579997062683, 0.2467200607061386, 0.2103690654039383, 0.23187917470932007, 
        0.28766655921936035, 0.21596665680408478, 0.24378667771816254, 0.2806374728679657, 
        0.23764009773731232, 0.2884339392185211, 0.19776469469070435, 0.29654744267463684, 
        0.23793953657150269, 0.2753768265247345, 0.24718035757541656, 0.2166261523962021, 
        0.22458019852638245, 0.36707887053489685, 0.29586368799209595, 0.24396133422851562, 
        0.3916597068309784, 0.2478819191455841, 0.3140171468257904, 0.23574240505695343, 
        0.30935078859329224, 0.2633970379829407, 0.22616524994373322, 0.22482863068580627, 
        0.25680482387542725, 0.184458926320076, 0.31002628803253174, 0.2936173677444458, 
        0.2688758671283722, 0.2438362091779709, 0.17232654988765717, 0.1869594156742096
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

