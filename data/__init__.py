from .realbsr import RealBSRDataset, RealBSRTextDataset, DF2KDataset, DIV2KDataset

def get_dataset(name, data_dir, **kwargs):
    if name == 'SyntheticBurstDF2K':
        return DF2KDataset(data_dir, **kwargs)
    elif name == 'SyntheticBurstDIV2K':
        return DIV2KDataset(data_dir, **kwargs)
    elif name == 'RealBSR':
        return RealBSRDataset(data_dir, **kwargs)
    elif name == 'RealBSR_text':
        return RealBSRTextDataset(data_dir, **kwargs)
    else:
        raise NotImplementedError()
