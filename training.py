from pretext_tasks.similari_pick import similari_pick
from pretext_tasks.RP import relativepositioning


def training(windows_dataset):
    """The training of the pretext tasks happens here."""
    emb,splitted1 = relativepositioning(windows_dataset)
    emb2, splitted2 = similari_pick(windows_dataset)
    return emb, emb2, splitted1, splitted2
