from functools import partial
import numpy as np
from braindecode.preprocessing.preprocess import (
    Preprocessor)
from braindecode.preprocessing.windowers import _create_windows_from_events

from utils.datasets import PC18


def scale(x, k):
    return k * x


def cast(x, dtype):
    return x.astype(dtype)


# this code is adapted from https://github.com/hubertjb/dynamic-spatial-filtering/blob/67a04b5a29b00564c6f3c9372f84f7f3fab2942f/utils.py#L59
def load_data(dataset, window_size_s, preload=None, save=None):
    """Load, preprocess and window data.
    """

    if dataset.startswith('pc18'):
        if dataset == 'pc18_debug':
            subject_ids = [989, 990, 991]
        elif dataset == 'pc18_hundred_files':
            subject_ids = range(989, 1089)
        else:
            subject_ids = 'training'

        # subject_ids = range(n_recs1, n_recs2)
        ch_names = ['F3-M2', 'F4-M1']
        preproc = [
            Preprocessor('pick_channels', ch_names=ch_names, ordered=True),
            Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=1),
            Preprocessor('resample', sfreq=100., n_jobs=1),
            Preprocessor(scale, k=1e6),
            Preprocessor(cast, dtype=np.float32)
        ]

        window_size_samples = int(window_size_s * 100)
        mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
        windower = partial(
            _create_windows_from_events, infer_mapping=False,
            infer_window_size_stride=False, trial_start_offset_samples=0,
            trial_stop_offset_samples=0, preload=preload,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples, mapping=mapping, accepted_bads_ratio=0.05, on_missing='warn')

        dataset = PC18(subject_ids=subject_ids, preproc=preproc,
                       windower=windower, n_jobs=1, save_dir=save)

    else:
        raise NotImplementedError

    return dataset
