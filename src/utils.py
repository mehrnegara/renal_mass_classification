import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss, loss_name, savedir, n_epoch):
    np.savetxt(f"{savedir}/{loss_name}.txt", loss)
    fig, ax1 = plt.subplots()
    ax1.plot(loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1, n_epoch])
    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png", dpi=200)    
    plt.close()           

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_OvR_AUC(true_labels, logits, num_classes):
    '''
    This function computes the One-vs-Rest (OvR) AUC for multi-class classification.
    
    Args:
        logits (np.ndarray): Probabilities of shape (n_samples, 1)
        true_labels (np.ndarray): True labels of shape (n_samples, 1)
        num_classes (int): Number of classes
        
    Returns:
        auc (np.ndarray): AUC for each class of shape (n_classes,)
    '''
    
    # Convert classes to one-hot encoding
    enc = OneHotEncoder()
    enc.fit(true_labels)
    true_ = enc.transform(true_labels).toarray()
    
    auc = np.zeros(num_classes)
    for i in range(num_classes):
        auc[i] = roc_auc_score(true_[:, i], logits[:, i])
    return auc
