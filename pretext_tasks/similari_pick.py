#author: Sergio Kazatzidis
def similari_pick(windows_dataset):
    """In this function the task of Frequency Similarity is performed and the features are extracted.
       This task takes an anchor window and two other windows. They are all transformed to the frequency domain
       with the Welch method. If the second window is more similar to the anchor window a label of 0 will be given,
       else a label of 1.

       Parameters
       ----------
       windows_dataset,
            a dataset of 30s windows of EEG waves.
       Output:
            Embeddings (features), splitted (SimilariPickDataset)
    """
    random_state = 87
    n_jobs = 1

    from scipy.signal._spectral_py import welch

    def freq_domain(p_before1, p_before2, q_before1, q_before2, z_before1,z_before2):
        #eeg-waves into frequency domain
        freqdomainp1 = welch(p_before1)
        freqdomainp2 = welch(p_before2)

        freqdomainq1 = welch(q_before1)
        freqdomainq2 = welch(q_before2)

        freqdomainz1 = welch(z_before1)
        freqdomainz2 = welch(z_before2)

        tot_sim1 = hellinger_explicit(freqdomainp1[1], freqdomainq1[1])
        tot_sim2 = hellinger_explicit(freqdomainp2[1], freqdomainq2[1])
        tot_sim3 = hellinger_explicit(freqdomainp1[1], freqdomainz1[1])
        tot_sim4 = hellinger_explicit(freqdomainp2[1], freqdomainz2[1])

        tot_sim = (tot_sim1 + tot_sim2) / 2
        tot_sim2 = (tot_sim3 + tot_sim4) / 2

        #give psuedo-labels
        if tot_sim < tot_sim2:
            return float(0)
        else:
            return float(1)

    import math
    def hellinger_explicit(p, q):
        """Hellinger distance between two discrete distributions.
           Same as original version but without list comprehension
        """
        list_of_squares = []
        for p_i, q_i in zip(p, q):
            # caluclate the square of the difference of ith distr elements
            s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

            # append
            list_of_squares.append(s)

        # calculate sum of squares
        sosq = sum(list_of_squares)

        return sosq / math.sqrt(2)

    ######################################################################
    # Splitting dataset into train, valid and test sets
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # We randomly split the recordings by subject into train, validation and
    # testing sets. We further define a new Dataset class which can receive a pair
    # of indices and return the corresponding windows. This will be needed when
    # training and evaluating on the pretext task.

    import numpy as np
    from sklearn.model_selection import train_test_split
    from braindecode.datasets import BaseConcatDataset
    sfreq = 100
    subjects = np.unique(windows_dataset.description['subject'])
    subj_train, subj_test = train_test_split(
        subjects, test_size=0.4, random_state=random_state)
    subj_valid, subj_test = train_test_split(
        subj_test, test_size=0.5, random_state=random_state)


    class SimilariPickDataset(BaseConcatDataset):
        """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
        """

        def __init__(self, list_of_ds):
            super().__init__(list_of_ds)
            self.return_pair = True

        def SimilarityTest(self, win_ind1, win_ind2,win_ind3):
            win_data1 = super().__getitem__(win_ind1)[0]
            win_data2 = super().__getitem__(win_ind2)[0]
            win_data3 = super().__getitem__(win_ind3)[0]
            y = freq_domain(win_data1[0],win_data1[1],win_data2[0],win_data2[1],win_data3[0],win_data3[1])
            return y

        def __getitem__(self, index):
            if self.return_pair:
                ind1, ind2,ind3, y = index
                x1 = super().__getitem__(ind1)[0]
                x2 = super().__getitem__(ind2)[0]
                x3 = super().__getitem__(ind3)[0]
                y_true = freq_domain(x1[0],x1[1],x2[0],x2[1],x3[0],x3[1])
                y_real = torch.from_numpy(np.array([y_true])).float()
                return (x1, x2, x3), y_real
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
        splitted[name] = SimilariPickDataset(
            [ds for ds in windows_dataset.datasets
             if ds.description['subject'] in values])

    ######################################################################
    # Creating samplers
    # ~~~~~~~~~~~~~~~~~

    from pretext_tasks.samplers import Similari_pickSampler

    n_examples_train = 2000 * len(splitted['train'].datasets)
    n_examples_valid = 2000 * len(splitted['valid'].datasets)
    n_examples_test = 2000 * len(splitted['test'].datasets)

    train_sampler = Similari_pickSampler(
        splitted['train'].get_metadata(),
        n_examples=n_examples_train, same_rec_neg=True, random_state=random_state)
    valid_sampler = Similari_pickSampler(
        splitted['valid'].get_metadata(),
        n_examples=n_examples_valid, same_rec_neg=True,
        random_state=random_state).presample()
    test_sampler = Similari_pickSampler(
        splitted['test'].get_metadata(),
        n_examples=n_examples_test, same_rec_neg=True,
        random_state=random_state).presample()


    ######################################################################
    # Creating the model
    # ------------------


    import torch
    from torch import nn
    from braindecode.util import set_random_seeds
    from braindecode.models import SleepStagerChambon2018

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
    # Set random seed to be able to roughly reproduce results
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
        emb : nn.Module
            Embedder architecture.
        emb_size : int
            Output size of the embedder.
        dropout : float
            Dropout rate applied to the linear layer of the contrastive module.
        """

        def __init__(self, emb, emb_size, dropout=0.5):
            super().__init__()
            self.emb = emb
            self.clf = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2*emb_size, 1)
            )

        def forward(self, x):
            x1, x2, x3 = x
            z1, z2, z3 = self.emb(x1), self.emb(x2), self.emb(x3)
            return self.clf(torch.cat((torch.abs(z1 - z2), torch.abs(z2 - z3)), dim=-1))
    model = ContrastiveNet(emb, emb_size).to(device)

    ######################################################################
    # Training
    # ---------
    #
    # We can now train our network on the pretext task.

    import os

    from skorch.helper import predefined_split
    from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
    from braindecode import EEGClassifier

    lr = 5e-4
    batch_size = 256
    n_epochs = 70
    num_workers = 0 if n_jobs <= 1 else n_jobs

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

    # comment lines below to use the saved models.
    #clf.initialize()
    #clf.load_params(f_params='saved_models/FS_params.pickle', f_optimizer='saved_models/FS_optimizer.pickle',
    #               f_history='saved_models/FS_history.pickle')


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
    plt.title("Sim-pick training")
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
    y_true = [y for _, _, _, y in test_sampler]

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    return emb, splitted

