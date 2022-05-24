from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

def get_camelyon17_loaders(args):
    kwargs ={}# {'root': args.data_root, 'split_scheme': args.camelyon17_split_scheme,
              #  'return_labels': True, 'return_meta_data': False, 'short_epoch': args.short_epoch}

    train_data = CivilCommentsDataset(split='train', **kwargs)
    val_data = CivilCommentsDataset(split='val', **kwargs)
    test_data = CivilCommentsDataset(split='test', **kwargs)

    print(train_data.dataset_name)

get_camelyon17_loaders(None)