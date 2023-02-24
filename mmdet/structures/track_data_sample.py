# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from .det_data_sample import DetDataSample


class TrackDataSample(BaseDataElement):

    @property
    def video_data_samples(self) -> List[DetDataSample]:
        return self._video_data_samples

    @video_data_samples.setter
    def video_data_samples(self, value: List[DetDataSample]):
        if isinstance(value, DetDataSample):
            value = [value]
        assert isinstance(value, list), 'video_data_samples must be a list'
        assert isinstance(
            value[0], DetDataSample
        ), 'video_data_samples must be a list of DetDataSample, but got '
        f'{value[0]}'
        self.set_field(value, '_video_data_samples', dtype=list)

    @video_data_samples.deleter
    def video_data_samples(self):
        del self._video_data_samples

    def __getitem__(self, index):
        assert hasattr(self,
                       '_video_data_samples'), 'video_data_samples not set'
        return self._video_data_samples[index]

    def get_key_frames(self):
        assert hasattr(self, 'key_frames_inds'), \
            'key_frames_inds not set'
        assert isinstance(self.key_frames_inds, Sequence)
        key_frames_info = []
        for index in self.key_frames_inds:
            key_frames_info.append(self[index])
        return key_frames_info

    def get_ref_frames(self):
        assert hasattr(self, 'ref_frames_inds'), \
            'ref_frames_inds not set'
        ref_frames_info = []
        assert isinstance(self.ref_frames_inds, Sequence)
        for index in self.ref_frames_inds:
            ref_frames_info.append(self[index])
        return ref_frames_info

    def __len__(self):
        return len(self._video_data_samples) if hasattr(
            self, '_video_data_samples') else 0

    # TODO: add UT for this Tensor-like method
    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if hasattr(v, 'to'):
                    v = v.to(*args, **kwargs)
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.cpu()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.cuda()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def npu(self) -> 'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.npu()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.detach()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, (torch.Tensor, BaseDataElement)):
                    v = v.detach().cpu().numpy()
                    data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v_list in self.items():
            data_list = []
            for v in v_list:
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                elif isinstance(v, BaseDataElement):
                    v = v.to_tensor()
                data_list.append(v)
            if len(data_list) > 0:
                new_data.set_data({f'{k}': data_list})
        return new_data


TrackSampleList = List[TrackDataSample]
OptTrackSampleList = Optional[TrackSampleList]
