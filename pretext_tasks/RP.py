

#code based on https://braindecode.org/stable/auto_examples/plot_relative_positioning.html#sphx-glr-auto-examples-plot-relative-positioning-py
# License: BSD (3-clause)


def relativepositioning(windows_dataset):
    """In this function the task of Relative Positioning[1] is performed and the features are extracted.
    This task takes an anchor window and a second window. If the second window is closeby it will give a label 1
    else it will give the label 0.

    Parameters
       ----------
       windows_dataset,
            a dataset of 30s windows of EEG waves.
    Output:
        Embeddings (features), splitted (RelativePositioningDataset)

    References
    ----------
    .. [Banville2020] Banville, H., Chehab, O., Hyvärinen, A., Engemann, D. A.,
           & Gramfort, A. (2020). Uncovering the structure of clinical EEG
           signals with self-supervised learning.
           arXiv preprint arXiv:2007.16104.
    """
    random_state = 87
    n_jobs = 1

    # Splitting dataset into train, valid and test sets

    import numpy as np
    from sklearn.model_selection import train_test_split
    from braindecode.datasets import BaseConcatDataset
    sfreq = 100
    subjects = np.unique(windows_dataset.description['subject'])
    subj_train, subj_test = train_test_split(
        subjects, test_size=0.4, random_state=random_state)
    subj_valid, subj_test = train_test_split(
        subj_test, test_size=0.5, random_state=random_state)


    class RelativePositioningDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """
        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)
            self.return_pair = True

        def __getitem__(self, index):
            if self.return_pair:
                ind1, ind2, y = index
                return (super().__getitem__(ind1)[0],
                        super().__getitem__(ind2)[0]), y
            else:
                return super().__getitem__(index)

        @property
        def return_pair(self):
            return self._return_pair

        @return_pair.setter
        def return_pair(self, value):
            self._return_pair = value


    split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
    splitted = dict()
    for name, values in split_ids.items():
        splitted[name] = RelativePositioningDataset(
            [ds for ds in windows_dataset.datasets
             if ds.description['subject'] in values])


    ######################################################################
    # Creating samplers
    # ~~~~~~~~~~~~~~~~~
    # The samplers control the number of pairs to be sampled (defined with
    # `n_examples`). 250 can be used if needing to debug.

    from pretext_tasks.samplers import RelativePositioningSampler

    tau_pos, tau_neg = int(sfreq * 60), int(sfreq * 15 * 60)
    n_examples_train = 2000 * len(splitted['train'].datasets)
    n_examples_valid = 2000 * len(splitted['valid'].datasets)
    n_examples_test = 2000 * len(splitted['test'].datasets)

    train_sampler = RelativePositioningSampler(
        splitted['train'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_train, same_rec_neg=True, random_state=random_state)
    valid_sampler = RelativePositioningSampler(
        splitted['valid'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_valid, same_rec_neg=True,
        random_state=random_state).presample()
    test_sampler = RelativePositioningSampler(
        splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_test, same_rec_neg=True,
        random_state=random_state).presample()


    ######################################################################
    # Creating the model
    # ------------------
    #
    # We can now create the deep learning model. In this tutorial, we use a
    # modified version of the sleep staging architecture introduced in [4]_ -
    # a four-layer convolutional neural network - as our embedder.
    # We change the dimensionality of the last layer to obtain a 100-dimension
    # embedding, use 16 convolutional channels instead of 8, and add batch
    # normalization after both temporal convolution layers.
    #
    # We further wrap the model into a siamese architecture using the
    # # :class:`ContrastiveNet` class defined below. This allows us to train the
    # feature extractor end-to-end.

    import torch
    from torch import nn
    from braindecode.util import set_random_seeds
    from braindecode.models import SleepStagerChambon2018

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    set_random_seeds(seed=random_state, cuda=device == 'cuda')

    # Extract number of channels and time steps from dataset
    n_channels, input_size_samples = windows_dataset[0][0].shape
    emb_size = 100

    emb = SleepStagerChambon2018(
        n_channels,
        sfreq,
        n_classes=emb_size,
        n_conv_chs=16,
        input_size_s=input_size_samples / sfreq,
        dropout=0,
        apply_batch_norm=True
    )


    class ContrastiveNet(nn.Module):
        """Contrastive module with linear layer on top of siamese embedder.

        Parameters
        ----------
        emb
            Embedder architecture.
        emb_size
            Output size of the embedder.
        dropout
            Dropout rate applied to the linear layer of the contrastive module.
        """
        def __init__(self, emb, emb_size, dropout=0.5):
            super().__init__()
            self.emb = emb
            self.clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(emb_size, 1)
            )

        def forward(self, x):
            x1, x2 = x
            z1, z2 = self.emb(x1), self.emb(x2)
            return self.clf(torch.abs(z1 - z2)).flatten()


    model = ContrastiveNet(emb, emb_size).to(device)


    ######################################################################
    # Training
    # ---------

    import os

    from skorch.helper import predefined_split
    from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
    from braindecode import EEGClassifier

    lr = 5e-4
    batch_size = 256
    n_epochs = 70
    num_workers = 0 if n_jobs <= 1 else n_jobs
    #if you want to save the model's parameters adjust below parameters to a place where to save them
    cp = Checkpoint(dirname='', f_criterion=None, f_optimizer=None, f_history=None)
    early_stopping = EarlyStopping(patience=10)
    train_acc = EpochScoring(
        scoring='accuracy', on_train=True, name='train_acc', lower_is_better=False)
    valid_acc = EpochScoring(
        scoring='accuracy', on_train=False, name='valid_acc',
        lower_is_better=False)
    callbacks = [
        ('cp', cp),
        ('patience', early_stopping),
        ('train_acc', train_acc),
        ('valid_acc', valid_acc)
    ]

    clf = EEGClassifier(
        model,
        criterion=torch.nn.BCEWithLogitsLoss,
        optimizer=torch.optim.Adam,
        max_epochs=n_epochs,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_sampler,
        iterator_valid__sampler=valid_sampler,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
        train_split=predefined_split(splitted['valid']),
        optimizer__lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device
    )
    # Model training for a specified number of epochs. `y` is None as it is already
    # supplied in the dataset.
    clf.fit(splitted['train'], y=None) # If you want to use the saved models, comment these two lines
    clf.load_params(checkpoint=cp)  # Load the model with the lowest valid_loss

    #comment lines below to use the saved models.
    #clf.initialize()
    #clf.load_params(f_params = 'saved_models/RP_params.pickle', f_optimizer = 'saved_models/RP_optimizer.pickle', f_history= 'saved_models/RP_history.pickle')



    #os.remove('params.pt')  # Delete parameters file


    ######################################################################
    # Visualizing the results
    # -----------------------
    #
    # Inspecting pretext task performance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # We plot the loss and pretext task performance for the training and validation
    # sets.

    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract loss and balanced accuracy values for plotting from history object
    df = pd.DataFrame(clf.history.to_list())

    df['train_acc'] *= 100
    df['valid_acc'] *= 100

    ys1 = ['train_loss', 'valid_loss']
    ys2 = ['train_acc', 'valid_acc']
    styles = ['-', ':']
    markers = ['.', '.']

    plt.style.use('seaborn-talk')

    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax2 = ax1.twinx()
    for y1, y2, style, marker in zip(ys1, ys2, styles, markers):
        ax1.plot(df['epoch'], df[y1], ls=style, marker=marker, ms=7,
                 c='tab:blue', label=y1)
        ax2.plot(df['epoch'], df[y2], ls=style, marker=marker, ms=7,
                 c='tab:orange', label=y2)

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Accuracy [%]', color='tab:orange')
    ax1.set_xlabel('Epoch')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    plt.title("RP training")
    plt.tight_layout()
    plt.show()


    ######################################################################
    # We also display the confusion matrix and classification report for the
    # pretext task:

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # Switch to the test sampler
    clf.iterator_valid__sampler = test_sampler
    y_pred = clf.forward(splitted['test'], training=False) > 0
    y_true = [y for _, _, y in test_sampler]

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return emb, splitted


