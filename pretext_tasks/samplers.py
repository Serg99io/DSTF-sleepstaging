import numpy as np
from braindecode.samplers import RecordingSampler


class RelativePositioningSampler(RecordingSampler):
    # from braindecode with some adjustments.
    """Sample examples for the relative positioning task from [Banville2020]_.

    Sample examples as tuples of two window indices, with a label indicating
    whether the windows are close or far, as defined by tau_pos and tau_neg.

    Parameters
    ----------
    metadata : pd.DataFrame
        See RecordingSampler.
    tau_pos : int
        Size of the positive context, in samples. A positive pair contains two
        windows x1 and x2 which are separated by at most `tau_pos` samples.
    tau_neg : int
        Size of the negative context, in samples. A negative pair contains two
        windows x1 and x2 which are separated by at least `tau_neg` samples and
        at most `tau_max` samples. Ignored if `same_rec_neg` is False.
    n_examples : int
        Number of pairs to extract.
    tau_max : int | None
        See `tau_neg`.
    same_rec_neg : bool
        If True, sample negative pairs from within the same recording. If
        False, sample negative pairs from two different recordings.
    random_state : None | np.RandomState | int
        Random state.

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
           & Gramfort, A. (2020). Uncovering the structure of clinical EEG
           signals with self-supervised learning.
           arXiv preprint arXiv:2007.16104.
    """

    def __init__(self, metadata, tau_pos, tau_neg, n_examples, tau_max=None,
                 same_rec_neg=True, random_state=None):
        super().__init__(metadata, random_state=random_state)

        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.tau_max = np.inf if tau_max is None else tau_max
        self.n_examples = n_examples
        self.same_rec_neg = same_rec_neg

        if not same_rec_neg and self.n_recordings < 2:
            raise ValueError('More than one recording must be available when '
                             'using across-recording negative sampling.')

    def _sample_pair(self):
        """Sample a pair of two windows.
        """
        mask = [0]
        count = 0
        while sum(mask) == 0:
            # Sample first window
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]['i_start_in_trial']
            ts = self.info.iloc[rec_ind1]['i_start_in_trial']

            # Decide whether the pair will be positive or negative
            pair_type = self.rng.binomial(1, 0.5)
            win_ind2 = None
            if pair_type == 0:  # Negative example
                if self.same_rec_neg:
                    mask = (
                            ((ts <= ts1 - self.tau_neg) & (ts >= ts1 - self.tau_max)) |
                            ((ts >= ts1 + self.tau_neg) & (ts <= ts1 + self.tau_max))
                    )
                else:
                    rec_ind2 = rec_ind1
                    while rec_ind2 == rec_ind1:
                        win_ind2, rec_ind2 = self.sample_window()
            elif pair_type == 1:  # Positive example
                mask = (ts >= ts1 - self.tau_pos) & (ts <= ts1 + self.tau_pos)

            if win_ind2 is None:
                mask[ts == ts1] = False  # same window cannot be sampled twice
                if count == 30:
                    # unless it is absolutely necessary
                    print("saaaaame")
                    return win_ind1, win_ind1, float(1)
                count = count + 1
        win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]['index'][mask])

        return win_ind1, win_ind2, float(pair_type)

    def presample(self):
        """Presample examples.

        Once presampled, the examples are the same from one epoch to another.
        """
        self.examples = [self._sample_pair() for _ in range(self.n_examples)]
        return self

    def __iter__(self):
        """Iterate over pairs.

        Yields
        ------
            (int): position of the first window in the dataset.
            (int): position of the second window in the dataset.
            (float): 0 for negative pair, 1 for positive pair.
        """
        for i in range(self.n_examples):
            if hasattr(self, 'examples'):
                yield self.examples[i]
            else:
                yield self._sample_pair()

    def __len__(self):
        return self.n_examples





class Similari_pickSampler(RecordingSampler):
    """Sample examples for the Frequency similarity task,

        Sample examples as three random window indices. The pseudo label will be given in the
        Similari_pickDataset because of convenience.

        Parameters
        ----------
        metadata : pd.DataFrame
            See RecordingSampler.
        n_examples : int
            Number of trios to extract.
        same_rec_neg : bool
            If True, sample negative pairs from within the same recording. If
            False, sample negative pairs from two different recordings.
        random_state : None | np.RandomState | int
            Random state.
        """


    def __init__(self, metadata, n_examples,
                 same_rec_neg=True, random_state=None):
        super().__init__(metadata, random_state=random_state)

        self.n_examples = n_examples
        self.same_rec_neg = same_rec_neg

        if not same_rec_neg and self.n_recordings < 2:
            raise ValueError('More than one recording must be available when '
                             'using across-recording negative sampling.')

    def _sample_trio(self):
        """Sample a trio of three windows.
        """
        # Sample first window
        win_ind1, rec_ind1 = self.sample_window()
        ts1 = self.metadata.iloc[win_ind1]['i_start_in_trial']
        ts = self.info.iloc[rec_ind1]['i_start_in_trial']
        win_ind2, rec_ind2 = self.sample_window(rec_ind=rec_ind1)
        win_ind3, rec_ind3 = self.sample_window(rec_ind=rec_ind1)
        count = 0
        while len(ts) == 1 or len(ts) == 2 or (win_ind1 == win_ind2) or (win_ind1 == win_ind3) or (
                win_ind2 == win_ind3):
            win_ind1, rec_ind1 = self.sample_window()
            ts1 = self.metadata.iloc[win_ind1]['i_start_in_trial']
            ts = self.info.iloc[rec_ind1]['i_start_in_trial']
            win_ind2, rec_ind2 = self.sample_window(rec_ind=rec_ind1)
            win_ind3, rec_ind3 = self.sample_window(rec_ind=rec_ind1)
            count = count + 1
            if count == 30:
                break

        if count == 30:
            return win_ind1, win_ind1, win_ind1, float(0)
        if -1 < count < 30:
            return win_ind1, win_ind2, win_ind3, float(0)

    def presample(self):
        """Presample examples.

        Once presampled, the examples are the same from one epoch to another.
        """
        self.examples = [self._sample_trio() for _ in range(self.n_examples)]
        return self

    def __iter__(self):
        """Iterate over trios.

        Yields
        ------
            (int): position of the first window in the dataset.
            (int): position of the second window in the dataset.
            (int): position of the third window in the dataset.
            (float): y (the right y is given in the main sim_freq function.)
        """
        for i in range(self.n_examples):
            if hasattr(self, 'examples'):
                yield self.examples[i]
            else:
                yield self._sample_trio()

    def __len__(self):
        return self.n_examples
