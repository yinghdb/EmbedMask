# EmbedMask: Embedding Coupling for One-stage Instance Segmentation

This is repository is for the paper: EmbedMask: Embedding Coupling for One-stage Instance Segmentation;


The full paper is available [here](https://www.ijcai.org/proceedings/2021/0175.pdf). 

## Installation
This EmbedMask implementation is based on [FCOS](https://github.com/tianzhi0549/FCOS), which is also based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). 
Therefore the installation is the same. Please check [INSTALL.md](https://github.com/tianzhi0549/FCOS/blob/master/INSTALL.md) of FCOS for installation instructions.

## Data Preparation

We do training and inference in the COCO dataset. If you want perform training and inference as well, please download the [dataset](http://cocodataset.org/#download), and put the data in the right place like follows.

    mkdir -p datasets/coco
    ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
    ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
    ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
    ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014

## Pretrained Models

The pretrained models can be downloaded from [here](https://1drv.ms/u/s!Al_gruIFwTUskAC9jf6oqkQ860of?e=3fHO6a). And you should place them in the 'models' directory.

## Demo

Once you have finished the installation and downloaded the pretrained models, you can run a quick demo by the following instructions which use the settings and from 'embed_mask_R50_1x'.
    
    # assume that you are under the root directory of this project,
    mkdir -p demo/output
    python demo/embed_mask_demo.py \
        --config-file configs/embed_mask/embed_mask_R50_1x.yaml \
        --weights models/embed_mask_R50_1x.pth


## Inference

The following inference command line run inference on coco minival split:

    CUDA_VISIBLE_DEVICES=0 \
    python tools/test_net.py \
        --config-file configs/embed_mask/embed_mask_R50_1x.yaml \
        MODEL.WEIGHT models/embed_mask_R50_1x.pth \
        TEST.IMS_PER_BATCH 4 \
        OUTPUT_DIR "./Results/R50-ConvAdd"

## Speed Testing

The following inference command line run speed testing on coco minival split:

    python tools/test_net.py \
        --config-file configs/embed_mask/embed_mask_R50_1x.yaml \
        --speed_only \
        MODEL.WEIGHT models/embed_mask_R50_1x.pth \
        TEST.IMS_PER_BATCH 1 \
        MODEL.EMBED_MASK.POSTPROCESS_MASKS True

## Training

The following command line will train 'embed_mask_R50_1x' on 4 GPUs with batchsize 16:

    CUDA_VISIBLE_DEVICES=0,2 \
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/embed_mask/embed_mask_R50_1x.yaml \
        DATALOADER.NUM_WORKERS 4 \
        SOLVER.IMS_PER_BATCH 8 \
        OUTPUT_DIR training_dir/embed_mask_R50_1x

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{ijcai2021-175,
  title     = {EmbedMask: Embedding Coupling for Instance Segmentation},
  author    = {Ying, Hui and Huang, Zhaojin and Liu, Shu and Shao, Tianjia and Zhou, Kun},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {1266--1273},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/175},
  url       = {https://doi.org/10.24963/ijcai.2021/175},
}

```
