# Extract features with the trained embedder
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def combine_emb(saved="no", file = None, emb1=None, splitted1=None, emb2 = None, useboth=None):
    """This function combines the embeddings of two pretext tasks by concatenation or you can choose to just use one pretext task. Afterwards it will test them on
    the downstream task sleep staging. Lastly, a UMAP visualization is used.

    Parameters
    ----------
    emb1,
        embedding one
    emb2,
        embedding two
    useboth,
        which indicates which embedding to use,
    saved,
        whether a saved file is present.
    Output:
        data (features)
    """
    import numpy as np
    batch_size = 128
    n_jobs = 1
    num_workers = 0 if n_jobs <= 1 else n_jobs
    data = dict()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    #if the features are not saved they will be put into the right format first.
    if saved == "no":

        for name, split in splitted1.items():
            split.return_pair = False  # Return single windows
            loader = DataLoader(split, batch_size=batch_size, num_workers=num_workers)
            with torch.no_grad():
                feats = [emb1(batch_x.to(device)).cpu().numpy()
                         for batch_x, _, _ in loader]
                feats2 = [emb2(batch_x.to(device)).cpu().numpy()
                          for batch_x, _, _ in loader]

            if useboth == "concat":
                new_featss = feats
                for i in range(len(feats)):
                    new_feat = feats[i]
                    after_concat = np.concatenate((new_feat, feats2[i]), axis=1)
                    new_featss[i] = after_concat

                data[name] = (np.concatenate(new_featss), split.get_metadata()['target'].values)

            if useboth == "1":
                data[name] = (np.concatenate(feats), split.get_metadata()['target'].values)
            if useboth == "2":
                data[name] = (np.concatenate(feats2), split.get_metadata()['target'].values)
        #here the features are put into pickle file.
        import pickle
        name = useboth + "-name" + ".pickle"
        with open(name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data
    if saved == "yes":
        import pickle
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data

def getting_results(amount, data, name = None):
    """This function gets the result of the downstream task with a certain amount of labeled data examples available. Afterward an Umap-visualization,"""
    random_state = 87
    log_reg = LogisticRegression(
        penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs',
        multi_class='multinomial', random_state=random_state)

    clf_pipe = make_pipeline(StandardScaler(), log_reg)
    import random
    if amount != "all":
        #pick an amount of random samples of each the classes [W,N1,N2,N3, R)
        lstclasses = [[] for _ in range(5)]
        for i in range(len(data["train"][1])):
            lstclasses[data["train"][1][i]].append(i)

        # we will use 87-91
        random_state = 87
        random.seed(random_state)
        together = None
        #sample randomly for each class
        for i in range(5):
            samp = random.sample(lstclasses[i],amount)
            if together == None:
                together = samp
            else:
                together = together + samp


        import sklearn.utils
        rand_lst = sklearn.utils.shuffle(together, random_state=random_state)
        new_lst = [data["train"][0][x] for x in rand_lst]
        new_lst2 = [data["train"][1][x] for x in rand_lst]
        #data with amount of random samples of each class
        new_data = (new_lst, new_lst2)
    data3 = dict()
    if amount == "all":
        data3["train"] = data["train"]
    else:
        data3["train"] = new_data
    data3["valid"] = data["valid"]
    data3["test"] = data["test"]
    clf_pipe.fit(*data3['train'])

    train_y_pred = clf_pipe.predict(data3['train'][0])
    valid_y_pred = clf_pipe.predict(data3['valid'][0])
    test_y_pred = clf_pipe.predict(data3['test'][0])
    train_bal_acc = balanced_accuracy_score(data3['train'][1], train_y_pred)
    valid_bal_acc = balanced_accuracy_score(data3['valid'][1], valid_y_pred)
    test_bal_acc = balanced_accuracy_score(data3['test'][1], test_y_pred)
    # print('Sleep staging performance with spatial shuffling with logistic regression:')
    print(f'Train bal acc: {train_bal_acc:0.4f}')
    print(f'Valid bal acc: {valid_bal_acc:0.4f}')
    print(f'Test bal acc: {test_bal_acc:0.4f}')

    print('Results on test set logistic regression:')
    print(confusion_matrix(data3['test'][1], test_y_pred))
    print(classification_report(data3['test'][1], test_y_pred))
    from matplotlib import cm

    X = np.concatenate([v[0] for k, v in data3.items()])
    y = np.concatenate([v[1] for k, v in data3.items()])

    # from sklearn.manifold import TSNE
    import umap
    # tsne = TSNE(n_components=2,perplexity=5)
    reducer = umap.UMAP(n_components=2)

    allvisual = [reducer]

    import seaborn as sns
    #umap visualizations are only applied when all the examples are used. this can be changed by not using this if statement
    if amount == "all":
        for j in range(len(allvisual)):
            model = allvisual[j]
            components = model.fit_transform(X)
            colors = cm.get_cmap('viridis', 5)(range(5))
            for i, stage in enumerate(['W', 'N1', 'N2', 'N3', 'R']):
                #the umap visualizations are done seperately since plotted together would lead to too much overlap.
                mask = y == i
                fig, ax = plt.subplots()
                ax.scatter(components[mask, 0], components[mask, 1], s=5, alpha=0.7,
                           color=colors[i], label=stage)
                ax.legend(markerscale=2)
                sns.kdeplot(x=components[:, 0], y=components[:, 1], color="black")
                if name == None:
                    plt.title("Pretext task UMAP")
                else:
                    plt.title(name)
                plt.xlabel("UMAP 1")
                plt.ylabel("UMAP 2")
                plt.show()


