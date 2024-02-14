import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import hydra

from joblib import dump
from scipy.io import loadmat
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset
from umap import UMAP

from neurorobotics_dl.events import Event
from neurorobotics_dl.metric_learning import (PrototypicalModel,
                                              get_all_embeddings, train)
from neurorobotics_dl.metric_learning.sampler import (BaseSampler,
                                                      EpisodicSampler)
from neurorobotics_dl.metric_learning.visualization import compare_embeddings
from neurorobotics_dl.models import EEGNet
from neurorobotics_dl.utils import fix_mat, summary

class MyDataset(Dataset):
    def __init__(self,X,y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return len(self.X)

"""________________________________________(EDITABLE) EXPERIMENT OPTIONS______________________________________"""

# RANDOM_SEED = 42 # Set it to None to implement fully random behaviour #TODO IMPLEMENT THIS COME ON!

# ## Data Processing options
# now = datetime.now().strftime('%Y%m%d_%H%M%S')

# DATA_PATH = f'C:/Users/tomma/Documents/Uni/PhD/data/cBCI/sbj4_tr_rigorous/preprocessed'
# OUTPUT_PATH = f'C:/Users/tomma/Documents/Uni/PhD/code/BCI-MI-bhbf/model/sbj4_tr_rigorous_{now}'
# LOG_PATH = f'C:/Users/tomma/Documents/Uni/PhD/code/BCI-MI-bhbf/logs/sbj4_tr_rigorous_{now}'

# SAMPLE_FREQUENCY = 512
# WIN_SIZE_S = 1
# WIN_SHIFT_S = 0.0625

# LABEL_MAP = { 0:'Both Hands',1:'Both Feets'} # Optional mapping (after dropping unwanted labels)

# NORMALIZE = 'zscore' # Data normalization applied at the end of the processing. Available options are 'zscore', None

# ## Model options

# DEVICE = 'cuda'

# # TGCN Params
# GCN_HIDDEN_DIMS = [128]
# GCN_OUTPUT_DIM = 16
# GCN_ACTIVATION = F.leaky_relu
# GRU_HIDDEN_UNITS = 128
# GCN_DROPOUT = 0.5
# GRU_DROPOUT = 0.3

# # EEGNet Params
# CONV_DROPOUT = 0.5
# F1 = 8
# D = 2
# F2 = 16
# EMBEDDING_DIM = 32

# ## Training options
# METRIC = 'cosine'
# LR = 1e-2
# ES_PATIENCE = 5
# ES_MIN_DELTA = 1e-4

# NUM_EPOCHS = 50
# EVAL_BATCH_SIZE = 128

# N_SUPPORT = 50
# N_QUERY = 60
# N_EPISODES = 500
# MAX_GRAD_NORM = 1
# NUM_CLASSES = 2

# LOG_INTERVAL = 1

# NUM_CHANNELS = 32
# OPTIMIZER = Adam
# OPTIMIZER_OPTIONS = {"lr":LR}
# SCHEDULER = LinearLR
# SCHEDULER_OPTIONS = {'start_factor':1,'end_factor':LR*1e-2,'total_iters':NUM_EPOCHS}

# USE_WANDB = False

def compute_distances(centroids,embeddings,metric):
    if metric == 'euclidean':
        _dist = [np.sum((embeddings - centroids[i])**2, dim=1) for i in range(centroids.shape[0])]
    else:
        _dist = [1-np.dot(embeddings, centroids[i])/(np.linalg.norm(embeddings)*np.linalg.norm(centroids[i]))for i in range(centroids.shape[0])]
    dist = np.stack(_dist,axis=1)
    return dist

@hydra.main(version_base="1.2",config_path="config", config_name="train")
def main(cfg=None):

    data_path = cfg.data_path
    output_dir = cfg.output_path
    log_dir = cfg.log_path
    fs = cfg.sample_frequency
    windowShift = int(cfg.win_size_s * fs)
    windowLen = int(cfg.sample_frequency * fs)
    nChannels = cfg.num_channels
    normalize = cfg.normalize
    label_map = cfg.label_map
    label_names = cfg.label_names

    device = cfg.train.device
    metric = cfg.train.metric
    learning_rate = cfg.train.lr
    num_epochs = cfg.train.num_epochs
    n_support = cfg.train.n_support
    n_query = cfg.train.n_query
    n_episodes = cfg.train.n_episodes
    n_classes = cfg.train.num_classes
    eval_batch_size = cfg.train.eval_batch_size
    max_grad_norm = cfg.train.max_grad_norm
    es_patience = cfg.train.es_patience
    es_min_delta = cfg.train.es_min_delta
    log_interval = cfg.train.log_interval
    use_wandb = cfg.train.use_wandb

    embedding_dim = cfg.EEGNET.embedding_dim
    dropout = cfg.EEGNET.conv_dropout
    f1 = cfg.EEGNET.f1
    d = cfg.EEGNET.d
    f2 = cfg.EEGNET.f2

    """________________________________________________PREPARE DATA_______________________________________________"""
    windows = dict()
    winlabels = dict()

    for split in ['train','val','test']:

        subj=loadmat(os.path.join(data_path,f'dataset_{split}.mat'))['subj']
        subj = fix_mat(subj) # fix errors while parsing.mat files

        eeg = np.expand_dims(subj['eeg'].T,1)
        triggers = subj['triggers']
        triggers['pos'] = triggers['pos'].astype(int)
        triggers['type'] = triggers['type'].astype(int)

        # Extract trials' start positions
        trial_starts = triggers['pos'][(triggers['type']==Event.CONT_FEEDBACK)]

        # Extract trials' end positions
        trial_ends = triggers['pos'][(triggers['type']==Event.HIT)|
                                    (triggers['type']==Event.MISS)|
                                    (triggers['type']==Event.TIMEOUT)]
        
        # Extract trials' labels
        trial_labels = triggers['type'][(triggers['type']==Event.REST)|
                                        (triggers['type']==Event.BOTH_FEET)|
                                        (triggers['type']==Event.BOTH_HANDS)]
        indices = []
        labels = []

        # Select trials of interest only
        mask = (trial_labels==Event.BOTH_FEET)|(trial_labels==Event.BOTH_HANDS)
        trial_starts = trial_starts[mask]
        trial_ends = trial_ends[mask]
        trial_labels = trial_labels[mask]

        # Compute trial indices
        nTrialsTot = 0
        for i in range(len(trial_starts)):
            t_start = trial_starts[i]
            t_end = trial_ends[i]
            assert t_start < t_end
            trial_idx = []
            while t_start + windowLen < t_end:
                trial_idx.append((t_start,t_start+windowLen))
                nTrialsTot +=1
                t_start += windowShift
            labels.append(label_map[trial_labels[i]]) ## FIXME per pigrizia non faccio mappe
            indices.append(trial_idx)
        num_features = 1
        windows[split] = np.zeros((nTrialsTot,nChannels,num_features,windowLen),dtype = np.float32)
        winlabels[split] = np.zeros(nTrialsTot,dtype = np.int64)

        seq = 0
        for i,idxlist in enumerate(indices):
            for start,end in idxlist:
                windows[split][seq,:,:] = eeg[:,:,start:end]
                winlabels[split][seq] = labels[i]
                seq+=1

    # Apply normalization
    if normalize is not None:
        if normalize=='zscore':
            mu = windows['train'].mean(axis=(0,3),keepdims=True)
            sigma = windows['train'].std(axis=(0,3),keepdims=True)

            windows['train'] = (windows['train']-mu)/sigma
            windows['val'] = (windows['val']-mu)/sigma
            windows['test'] = (windows['test']-mu)/sigma
        else:
            mu,sigma = 0,1

    # Wrap datasets in a custom object
    train_set = MyDataset(windows['train'],winlabels['train'])
    val_set = MyDataset(windows['val'],winlabels['val'])
    test_set = MyDataset(windows['test'],winlabels['test'])

    # Create samplers
    episodic_sampler = EpisodicSampler(train_set,
                                       n_support=n_support,
                                       n_query=n_query,
                                       n_episodes=n_episodes,
                                       n_classes=n_classes)
    train_sampler = BaseSampler(train_set, shuffle=False, batch_size=eval_batch_size)
    val_sampler = BaseSampler(val_set, batch_size=eval_batch_size)
    test_sampler = BaseSampler(test_set, shuffle=False, batch_size=eval_batch_size)

    """________________________________________________TRAIN MODEL________________________________________________"""

    ## Create model	

    # T-GCN
    # nChannels = 16
    # gcn_input_dim = 1
    # gcn_hidden_dims = GCN_HIDDEN_DIMS
    # gcn_output_dim = GCN_OUTPUT_DIM
    # gcn_activation = GCN_ACTIVATION
    # gru_hidden_units = GRU_HIDDEN_UNITS
    # gcn_dropout = GRU_DROPOUT
    # gru_dropout = GCN_DROPOUT
    # num_classes = NUM_CLASSES

    # net = GCN_GRU_sequence_fxdD(16,
    #                         gcn_input_dim,
    #                         gcn_hidden_dims,
    #                         gcn_output_dim,
    #                         gcn_activation,
    #                         gru_hidden_units,
    #                         gcn_dropout,
    #                         gru_dropout,
    #                         )

    ## EEGNet

    net = EEGNet(embedding_dim, 
                Chans = nChannels, 
                Samples = windowLen,
                dropoutRate = dropout,
                kernLength = windowLen//2,
                F1 = f1, 
                D = d, 
                F2 = f2,)

    model = PrototypicalModel(net,metric).to('cuda')

    summary(model)

    train(model,
        episodic_sampler,
        train_sampler,
        val_sampler,
        num_epochs,
        learning_rate=learning_rate,
        device=device,
        log_dir=log_dir,
        log_interval=log_interval,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        es_patience = es_patience,
        es_min_delta = es_min_delta,
        use_wandb = use_wandb,
        )

    """__________________________________________________RESULTS__________________________________________________"""

    model.load_state_dict(torch.load(os.path.join(output_dir,'model.pt')))

    model.to('cuda')
    train_embeddings, train_labels = get_all_embeddings(train_sampler, model,device)
    train_embeddings = train_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()

    val_embeddings, val_labels = get_all_embeddings(val_sampler, model,device)
    val_embeddings = val_embeddings.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    test_embeddings, test_labels = get_all_embeddings(test_sampler, model,device)
    test_embeddings = test_embeddings.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    tr_len = len(train_embeddings)
    val_len = tr_len+len(val_embeddings)
    all_embeddings = np.concatenate([train_embeddings,val_embeddings,test_embeddings])
    transform = UMAP(metric=metric)
    all_embeddings_umap = transform.fit_transform(all_embeddings)
    compare_embeddings(all_embeddings_umap[:tr_len],
                    all_embeddings_umap[tr_len:val_len],
                    train_labels,
                    val_labels,
                    embeddings3 = all_embeddings_umap[val_len:],
                    labels3=test_labels,label_mappings = label_names,
                    save_as = os.path.join(log_dir,"embeddings.png"))

    ## Try different classifiers
    print("\nDistance-based Classifier:")

    centroids = (train_embeddings[train_labels==0].mean(axis=0),train_embeddings[train_labels==1].mean(axis=0))
    centroids = np.stack(centroids)

    train_preds = compute_distances(centroids,train_embeddings,metric).argmin(axis=1)
    val_preds = compute_distances(centroids,val_embeddings,metric).argmin(axis=1)
    test_preds = compute_distances(centroids,test_embeddings,metric).argmin(axis=1)

    print(f'Train Accuracy: {(train_preds==train_labels).mean()*100:.2f}%')
    print(f'  Val Accuracy: {(val_preds==val_labels).mean()*100:.2f}%')
    print(f' Test Accuracy: {(test_preds==test_labels).mean()*100:.2f}%')

    ## Try different classifiers
    print("\nQuadratric Discriminant Analysis Classifier:")
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(train_embeddings, train_labels)

    train_preds = clf.predict(train_embeddings)
    val_preds = clf.predict(val_embeddings)
    test_preds = clf.predict(test_embeddings)
    print(f'Train Accuracy: {(train_preds==train_labels).mean()*100:.2f}%')
    print(f'  Val Accuracy: {(val_preds==val_labels).mean()*100:.2f}%')
    print(f' Test Accuracy: {(test_preds==test_labels).mean()*100:.2f}%')

    dump(clf, os.path.join(output_dir,'qda.joblib') )
    np.savez(os.path.join(output_dir,'mean_std.npz'),mu=mu,sigma=sigma)

    print("Computing kernels")
    train_probs = clf.predict_proba(train_embeddings)
    val_probs = clf.predict_proba(val_embeddings)
    test_probs = clf.predict_proba(test_embeddings)

    train_probs = clf.predict_proba(train_embeddings)
    val_probs = clf.predict_proba(val_embeddings)
    test_probs = clf.predict_proba(test_embeddings)

    sns.set_style('whitegrid')
    h_mu,h_std = train_probs[train_labels==0][:,0].mean(),train_probs[train_labels==0][:,0].std()
    f_mu,f_std = 1-train_probs[train_labels==1][:,1].mean(),train_probs[train_labels==1][:,1].std()

    sns.kdeplot(np.random.normal(h_mu,h_std,1000))
    sns.kdeplot(np.random.normal(f_mu,f_std,1000))
    plt.legend(['Both Hands','Both Feet'])
    plt.savefig(os.path.join(output_dir,"kernels.py"))

    print("End.")

if __name__ == '__main__':
    main()


