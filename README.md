# SA-ArcFace
SA-ArcFace: Self-Attention Enhanced ArcFace for Deep Face Recognition

## Requirements
In order to enjoy the new features of pytorch, we have upgraded the pytorch to 1.9.0.  
- tensorboard
- easydict
- mxnet
- onnx
- sklearn

## How to Training
To train a model, run `train.py` with the path to the configs.  
The example commands below show how to run
distributed training.

```shell
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms1mv2_r100.py
```
If you want to train on a machine with multiple GPUs, you can achieve this by `--nproc_per_node`. For example, on a machine with 8 GPUs:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/ms1mv2_r100.py
```

## Download Datasets or Prepare Datasets  
InsightFace provides a range of preprocessed labeled face datasets, including the MS1MV2 dataset used in SA_ArcFace.
- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)

If you want to try other datasets provided by InsightFace, you can quickly find them through the following link:
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)
