from utils.data_loader import load_data
from training import training
from combining_emb import combine_emb, getting_results

"""Here it is possible to go through each step until getting the results of the sleep staging downstream task. 
It is also possible to use the saved RP and FS models."""

"first step is loading and preprocessing the data. Possible to use pc18_debug for a smaller dataset."
windows_dataset = load_data('pc18', 30, 1)

"the second step is getting the features of the pretext tasks in training.py"
emb, emb2, splitted1, splitted2 = training(windows_dataset)

"the third step is combining the embeddings. Data are the features either from saved files or made."
data = combine_emb(saved="no", emb1=emb, splitted1=splitted1,emb2=emb2,useboth = "concat")
"use this below when you have the features saved"
#data = combine_emb(saved="yes",file="features/RP.pickle")

"the fourth step is getting the results and visualizations. The amount indicates the amount of labeled examples used for training."
getting_results(amount="all",data=data,name = "RP+FreqSim")

