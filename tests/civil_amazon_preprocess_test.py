import wilds
from requests import head
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import argparse

import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import sys
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
#from train import train, evaluate, infer_predictions
#from algorithms.initializer import initialize_algorithm, infer_d_out
import transformers as ppb
from transforms import initialize_transform
#from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

# they use config file to specify the dataset, algorithm, shceluder, optimizor, tranaform, model, etc.
#define the config file
def main():
    #get the configuartion:
    ''' Arg defaults are filled in according to ./configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    #parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    #parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    #parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

     # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')


    config = parser.parse_args()
    config = populate_defaults(config)

    # Load the full dataset, and download it if necessary
    full_dataset = get_dataset(  
                        dataset=config.dataset, #"civilcomments", 
                        version=config.version, #'1.0',
                        download=config.download, #False,
                        root_dir = config.root_dir # "D:/distributionShiftAmazonData/data/civilComments/"
                    )

    ## get the split info and write to a csv
    splits = pd.DataFrame(full_dataset.split_array)

    #write the train/eval/test splits into csv
    #splits.to_csv("./civilcommentsSplits.csv", header=False, index=False)

    print(full_dataset.split_array)
    print(len(full_dataset.split_array))
    print(full_dataset.split_dict)
    print(full_dataset.split_names)


    # Get the training set
    # transform is for data augumentation
    # For the amazon dataset, the split is already in the downloaded dataset: the user.csv

    '''
    # Data
    full_dataset = get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)
    '''

# Transforms & data augmentations for labeled dataset
    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    train_transform = initialize_transform(
                        transform_name=config.transform,
                        config=config,
                        dataset=full_dataset,
                        additional_transform_name=config.additional_train_transform,
                        is_training=True)

    eval_transform = initialize_transform(
                    transform_name=config.transform,
                    config=config,
                    dataset=full_dataset,
                    is_training=False)


    #take  fraction of the training dataset, and transform it using the bert algorithm
    np.random.seed(1)
    
    '''
    train_data = full_dataset.get_subset(
        "train",
        frac=1,
        transform=None, #train_transform,
        #do_transform_y=True,
    )
    print(len(train_data.split_array)) ##number of samples in the train data
    '''

    eval_data = full_dataset.get_subset(
        "val",
        frac=0.002,
        transform=eval_transform,
        #do_transform_y=True,
    )

    #print(eval_data['dataset'])
    

    # get transformed items
    print(len(eval_data.split_array))
    print("The indices for the data is: ")
    print(eval_data.indices)
    print(eval_data.dataset)
    #x,y,meta = eval_data.__getitem__(0)
    x, y, meta = eval_data.__get_all_transformed_items__()
    print(x[0])
    print(x.shape)
    print(y[0])
    print(meta[0])
    #print()#[0])
    
    #get all original items and may stored in local



    #put the tensors into the bert model and extract the features
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    ## Want BERT instead of distilBERT? Uncomment the following line:
    #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    #tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    input_ids, attention_mask = x[:,:,0], x[:,:,1]
    input_ids = torch.tensor(input_ids).to(torch.int64)
    attention_mask = torch.tensor(attention_mask).to(torch.int64)

    print(input_ids.shape)
    print(attention_mask.shape)
    model = model_class.from_pretrained(pretrained_weights)

    with torch.no_grad():
        last_hidden_states = model( input_ids, attention_mask=attention_mask )
    features = last_hidden_states[0][:,0,:].numpy()
    print("The features by bert is:")
    #print(features[0])
    f1 = pd.DataFrame(features)
    f1['class'] = y
    f1.to_csv("All features_tmp.csv")

    train_features, test_features, train_labels, test_labels = train_test_split(features, y)
    print(train_features.shape)
    print(test_features.shape)

    #lr_clf = LogisticRegression()
    #lr_clf.fit(train_features, train_labels)

    from sklearn.ensemble import RandomForestClassifier
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=200)
    clf.fit(train_features, train_labels)

    print( "The classification result is: %0.3f" % clf.score(test_features, test_labels) )

    # compare with a dummy classifier
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier()
    scores = cross_val_score(clf, train_features, train_labels)
    print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# bert transform function for transforming the text， should be in another library
def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


'''
# Prepare the standard data loader
#train_loader = get_train_loader("standard", train_data, batch_size=16)


# (Optional) Load unlabeled data
dataset = get_dataset(dataset="amazon", download=True, unlabeled=True)
unlabeled_data = dataset.get_subset(
    "test_unlabeled",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# Train loop
for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
    x, y, metadata = labeled_batch
    unlabeled_x, unlabeled_metadata = unlabeled_batch
    ...
'''

if __name__ == '__main__':
    main()