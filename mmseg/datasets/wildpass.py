from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class WildPassDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['car', 'truck', 'bus', 'road', 'sidewalk', 'person', 'curb', 'crosswalk'],  # background is not included
        palette=[[0, 0, 142], [0, 0, 70], [0, 60, 100], [128, 64, 128], [244, 35, 232], [220, 20, 60], [196, 196, 196], [200, 128, 128]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
