from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class IFLYInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['CultivatedLand'],
        'palette': [(255, 255, 0)]
    }

@DATASETS.register_module()
class AI4BInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['cropland_parcel'],
        'palette': [(0, 255, 0)]
    }

@DATASETS.register_module()
class CAPCADInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['CultivatedLand'],
        'palette': [(255, 255, 0)]
    }