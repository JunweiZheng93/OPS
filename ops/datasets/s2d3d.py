from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class S2D3DDataset(BaseSegDataset):

    METAINFO = dict(
        classes=['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window'],  # background is not included
        palette=[[216,86,98], [217,129,89], [148,108,9], [196,217,81], [71,154,24], [7,54,24], [91,114,29], [93,199,167], [52,121,177], [52,12,77], [154,81,170], [194,86,190], [94,86,90]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
