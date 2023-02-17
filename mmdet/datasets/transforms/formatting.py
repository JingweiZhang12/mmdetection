# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes


@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class ToTensor:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class Transpose:
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='Pad', size_divisor=32),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapFieldsToLists')
        >>> ]
    """

    def __call__(self, results):
        """Call function to wrap fields into lists.

        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict where value of ``self.keys`` are wrapped \
                into list.
        """

        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class PackTrackInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation.

    All the information of images are packed to ``inputs``.
    All the information except images are packed to ``data_samples``.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_id': 'instances_id'
    }

    def __init__(self,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'video_id',
                                             'video_length', 'instances')):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.transpose(0, 3, 1, 2)
            packed_results['inputs'] = to_tensor(imgs)

        # 2. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags_list = results['gt_ignore_flags']
            valid_idx_list, ignore_idx_list = [], []
            for gt_ignore_flags in gt_ignore_flags_list:
                valid_idx = np.where(gt_ignore_flags == 0)[0]
                ignore_idx = np.where(gt_ignore_flags == 1)[0]
                valid_idx_list.append(valid_idx)
                ignore_idx_list.append(ignore_idx)

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        num_imgs = len(results['img_id'])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]
        ignore_instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, gt_mask in enumerate(gt_masks_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_mask[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_mask[
                            ignore_idx]

                else:
                    for i, gt_mask in enumerate(gt_masks_list):
                        instance_data_list[i][mapped_key] = gt_mask

            else:
                anns_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, ann in enumerate(anns_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[valid_idx])
                        ignore_instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[ignore_idx])
                else:
                    for i, ann in enumerate(anns_list):
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_sample.ignored_instances = ignore_instance_data_list[i]
            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f'{key}': img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        if 'key_frame_inds' in results:
            track_data_sample.set_metainfo(
                dict(key_frame_inds=results['key_frame_inds']))
        if 'ref_frame_inds' in results:
            track_data_sample.set_metainfo(
                dict(ref_frame_inds=results['ref_frame_inds']))

        packed_results['data_samples'] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str
