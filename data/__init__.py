from .realbsr import RealBSRDataset, RealBSRTextDataset


def get_dataset(name, data_dir, **kwargs):
    if name == 'RealBSR':
        return RealBSRDataset(data_dir, **kwargs)
    elif name == 'RealBSR_text':
        return RealBSRTextDataset(data_dir, **kwargs)
    else:
        raise NotImplementedError()
