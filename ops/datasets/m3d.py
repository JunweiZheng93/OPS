from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class M3DDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects', 'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror', 'shower', 'bathtub', 'counter', 'shelving'],  # background is not included
        palette=[[174, 199, 232], [112, 128, 144], [152, 223, 138], [197, 176, 213], [255, 127,  14], [214,  39,  40],
                 [31, 119, 180], [188, 189,  34], [255, 152, 150], [44, 160,  44], [227, 119, 194], [132,  60,  57],
                 [158, 218, 229], [156, 158, 222], [231, 150, 156], [219, 219, 141], [206, 219, 156], [57, 59, 121],
                 [165,  81, 148], [196, 156, 148]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
