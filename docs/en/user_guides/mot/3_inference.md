# Inference

We provide demo scripts to inference a given video or a folder that contains continuous images. The source codes are available [here](https://github.com/open-mmlab/mmtracking/tree/dev-1.x/demo/).

Note that if you use a folder as the input, the image names there must be  **sortable** , which means we can re-order the images according to the numbers contained in the filenames. We now only support reading the images whose filenames end with `.jpg`, `.jpeg` and `.png`.

## Inference MOT models

This script can inference an input video / images with a multiple object tracking or video instance segmentation model.

```shell
python demo/demo_mot.py \
    ${CONFIG_FILE} \
    --input ${INPUT} \
    [--output ${OUTPUT}] \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--score-thr ${SCORE_THR} \
    [--device ${DEVICE}] \
    [--show]
```

The `INPUT` and `OUTPUT` support both _mp4 video_ format and the _folder_ format.

**Important:** For `DeepSORT`, `SORT`, `Tracktor`, `StrongSORT`, they need both the weight of the `reid` and the weight of the `detector`. Therefore, we can't use `--checkpoint` to specify it. We need to use `init_cfg` in the configuration file to set the weight path. Other algorithms such as `ByteTrack`, `OCSORT` and `QDTrack` need not pay attention to this.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `CHECKPOINT_FILE`: The checkpoint is optional in case that you already set up the pretrained models in the config by the key `init_cfg`.
- `SCORE_THR`: The threshold of score to filter bboxes.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `--show`: Whether show the video on the fly.

**Examples of running mot model:**

```shell
# Example 1: do not specify --checkpoint to use the default init_cfg
python demo/demo_mot.py \
    configs/sort/sort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --output mot.mp4

# Example 2: use --checkpoint
python demo/demo_mot.py \
    configs/qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py \
    --input demo/demo.mp4 \
    --checkpoint https://download.openmmlab.com/mmtracking/mot/qdtrack/mot_dataset/qdtrack_faster-rcnn_r50_fpn_4e_mot17_20220315_145635-76f295ef.pth \
    --output mot.mp4
```
