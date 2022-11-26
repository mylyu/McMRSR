from . import cartesian_dataset, cartesian_dataset_h5


def get_datasets(opts):
    if opts.dataset == 'M4Raw':
        trainset = cartesian_dataset_h5.MRIDataset_Cartesian(opts, mode='TRAIN')
        valset = cartesian_dataset_h5.MRIDataset_Cartesian(opts, mode='VALI')
        testset = cartesian_dataset_h5.MRIDataset_Cartesian(opts, mode='TEST')
    elif opts.dataset == 'Clinical':
        trainset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='TRAIN')
        valset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='VALI')
        testset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='TEST')
    else:
        raise NotImplementedError

    return trainset, valset, testset
