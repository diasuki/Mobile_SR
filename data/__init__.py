from .realbsr import RealBSRDataset, RealBSRTextDataset, DF2KDataset, DIV2KDataset

def get_dataset(name, data_dir, **kwargs):
    if name.startswith('SyntheticBurstDF2K'):
        return DF2KDataset(name, data_dir, **kwargs)
    elif name.startswith('SyntheticBurstDIV2K'):
        return DIV2KDataset(name, data_dir, **kwargs)
    elif name == 'RealBSR':
        return RealBSRDataset(data_dir, **kwargs)
    elif name == 'RealBSR_text':
        return RealBSRTextDataset(data_dir, **kwargs)
    else:
        raise NotImplementedError()
