# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from fcos_core.config import cfg
from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def compute_on_dataset(model, data_loader, device, timer=None, start_iter=0, break_iter=0, speed_only=False, benchmark=False, timers=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        if i < start_iter:
            results_dict.update(
                {img_id: [] for img_id in image_ids}
            )
            continue
        if break_iter > 0 and i == break_iter:
            break
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device), benchmark=benchmark, timers=timers)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
        if not (speed_only or benchmark):
            output = [o.to(cpu_device) for o in output]
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        start_iter=0,
        break_iter=0,
        speed_only=False,
        benchmark=False,
        cfg=None,
        multi_test=False
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    if break_iter == 0:
        break_iter = len(dataset)
    if benchmark:
        timers = [Timer() for i in range(11)]
    else: 
        timers = None
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer,
                                     start_iter=start_iter, break_iter=break_iter, 
                                     speed_only=speed_only, benchmark=benchmark, timers=timers)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / (break_iter - start_iter), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, fps {}, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / (break_iter - start_iter),
            (break_iter - start_iter) / (inference_timer.total_time * num_devices),
            num_devices,
        )
    )
    if benchmark:
        for i in range(len(timers)):
            logger.info("timer {}: {} s)".format(
                i, timers[i].total_time * num_devices / (break_iter - start_iter), )
            )
        return
        
    if speed_only or benchmark:
        return

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    if multi_test:
        margin = 0.45
        for i in range(10):
            print("########################################################")
            print("########################################################")
            print("margin %f\n" % (margin))
            for i in range(len(predictions)):
                predictions[i].add_field('mask_th', torch.tensor(margin))
            evaluate(dataset=dataset,
                     predictions=predictions,
                     output_folder=output_folder,
                     **extra_args)
            margin = margin + 0.01
        return
    else:
        return evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)