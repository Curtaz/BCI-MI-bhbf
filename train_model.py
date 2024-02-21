import os
import random

import numpy as np
import torch
import hydra

from joblib import dump
from scipy.io import loadmat
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from torch.utils.data import Dataset
from umap import UMAP
from omegaconf import OmegaConf

from neurorobotics_dl.events import Event
from neurorobotics_dl.metric_learning import (PrototypicalModel,
                                              get_all_embeddings, train)
from neurorobotics_dl.metric_learning.sampler import (BaseSampler,
                                                      EpisodicSampler)
from neurorobotics_dl.metric_learning.visualization import compare_embeddings
from neurorobotics_dl.utils import fix_mat, summary,write_to_yaml,get_class


def set_reproducibility(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

"""__________________________________________________UTILITIES________________________________________________"""

class MyDataset(Dataset):
    def __init__(self,X,y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return len(self.X)

def compute_distances(centroids,embeddings,metric):
    if metric == 'euclidean':
        _dist = [np.sum((embeddings - centroids[i])**2, dim=1) for i in range(centroids.shape[0])]
    else:
        _dist = [1-np.dot(embeddings, centroids[i])/(np.linalg.norm(embeddings)*np.linalg.norm(centroids[i]))for i in range(centroids.shape[0])]
    dist = np.stack(_dist,axis=1)
    return dist
   
"""____________________________________________________MAIN___________________________________________________"""
@hydra.main(version_base="1.2",config_path="config/model_training", config_name="train")
def main(cfg=None):

    random_seed = cfg.random_seed        
    data_path = cfg.data_path
    output_dir = cfg.output_path
    log_dir = cfg.log_path
    fs = cfg.sample_frequency
    windowShift = int(cfg.win_shift_s * fs)
    windowLen = int(cfg.win_size_s * fs)
    nChannels = cfg.num_channels
    normalize = cfg.normalize
    label_map = cfg.label_map
    label_names = cfg.label_names
    classes = cfg.classes

    pre_trained_model = cfg.train.pre_trained_model
    device = cfg.train.device
    metric = cfg.train.metric
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

    if random_seed is not None:
        set_reproducibility(random_seed)

    model_cl = get_class(cfg.model.classname)
    model_options = cfg.model.options

    optim_cl = get_class(cfg.train.optimizer.classname)
    optimizer_options = cfg.train.optimizer.options

    lr_scheduler_cl = get_class(cfg.train.scheduler.classname)
    lr_scheduler_options = cfg.train.scheduler.options

    """________________________________________________PREPARE DATA_______________________________________________"""
    windows = dict()
    winlabels = dict()
    filenames = dict()

    print("Preparing data")
    for split in ['train','val','test']:
        subj=loadmat(os.path.join(data_path,f'dataset_{split}.mat'))['subj']
        subj = fix_mat(subj) # fix errors while parsing.mat files
        filenames[split] = subj['filenames'].tolist()
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
        mask = np.zeros(len(trial_labels),dtype='bool')
        for cl in classes:
            mask = mask|(trial_labels==cl)   
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
            labels.append(label_map[int(trial_labels[i])])
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

        print(f"{split.capitalize()}: number of samples:",len(windows[split]))
        for label,name in label_names.items():
            print(f"\t{name}: {(winlabels[split]==label).sum()}")

    # Apply normalization
    print("Applying normalization:",normalize)
    if normalize is not None:
        if normalize=='zscore':
            mu = windows['train'].mean(axis=(0,3),keepdims=True)
            sigma = windows['train'].std(axis=(0,3),keepdims=True)

            windows['train'] = (windows['train']-mu)/sigma
            windows['val'] = (windows['val']-mu)/sigma
            windows['test'] = (windows['test']-mu)/sigma
        else: 
            raise ValueError(f"Normalization type {normalize} not supported. Supported types are 'zscore',None")
    else:
        mu,sigma = 0,1

    print("\nCreating samplers")
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

    """________________________________________CREATE MODEL AND OPTIMIZERS________________________________________"""
    print("Creating model and optimizers")
    net = model_cl(**model_options)
    model = PrototypicalModel(net,metric).to('cuda')
    if pre_trained_model is not None:
        model.load_state_dict(torch.load(pre_trained_model))
    summary(model)

    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = optim_cl(parameters,**optimizer_options)
    lr_scheduler = lr_scheduler_cl(optimizer,**lr_scheduler_options)

    """___________________________________________________TRAIN___________________________________________________"""

    print()
    
    train(model,
        episodic_sampler,
        train_sampler,
        val_sampler,
        num_epochs,
        optimizer = optimizer,
        scheduler = lr_scheduler,
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

    print("\nBeginning test phase")
    model.load_state_dict(torch.load(os.path.join(output_dir,'model.pt')))
    model.to(device)
    
    print("\nComputing train embeddings")
    train_embeddings, train_labels = get_all_embeddings(train_sampler, model,device)
    train_embeddings = train_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()

    print("Computing val embeddings")
    val_embeddings, val_labels = get_all_embeddings(val_sampler, model,device)
    val_embeddings = val_embeddings.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    print("Computing test embeddings")
    test_embeddings, test_labels = get_all_embeddings(test_sampler, model,device)
    test_embeddings = test_embeddings.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    print("\nResults:")
    ## Try different classifiers
    print("Distance-based Classifier:")
    centroids = (train_embeddings[train_labels==0].mean(axis=0),train_embeddings[train_labels==1].mean(axis=0))
    centroids = np.stack(centroids)
    train_preds = compute_distances(centroids,train_embeddings,metric).argmin(axis=1)
    val_preds = compute_distances(centroids,val_embeddings,metric).argmin(axis=1)
    test_preds = compute_distances(centroids,test_embeddings,metric).argmin(axis=1)
    print(f'  Train Accuracy: {(train_preds==train_labels).mean()*100:.2f}%')
    print(f'    Val Accuracy: {(val_preds==val_labels).mean()*100:.2f}%')
    print(f'   Test Accuracy: {(test_preds==test_labels).mean()*100:.2f}%')

    ## Try different classifiers
    print("Quadratric Discriminant Analysis Classifier:")
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(train_embeddings, train_labels)
    train_preds = clf.predict(train_embeddings)
    val_preds = clf.predict(val_embeddings)
    test_preds = clf.predict(test_embeddings)
    print(f'  Train Accuracy: {(train_preds==train_labels).mean()*100:.2f}%')
    print(f'    Val Accuracy: {(val_preds==val_labels).mean()*100:.2f}%')
    print(f'   Test Accuracy: {(test_preds==test_labels).mean()*100:.2f}%')

    write_to_yaml(os.path.join(output_dir,'config.yaml'),
                {'model': OmegaConf.to_object(cfg.model),
                 'classes': OmegaConf.to_object(classes),
                'label_map': OmegaConf.to_object(label_map),
                'filenames': filenames,
                })
    dump(clf, os.path.join(output_dir,'qda.joblib') )
    np.savez(os.path.join(output_dir,'mean_std.npz'),mu=mu,sigma=sigma)
    print(f"\nModel parameters and classifier saved at {output_dir}")

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
    print(f'Feature map saved at {os.path.join(log_dir,"embeddings.png")}')

    print("End.")

if __name__ == '__main__':
    main()


