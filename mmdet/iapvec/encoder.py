import torch
#import numpy as np
#from hisup.csrc.lib.afm_op import afm
from torch.utils.data.dataloader import default_collate

class Encoder(object):
    def __init__(self):
        self.target_h = 128
        self.target_w = 128

    def __call__(self, annotations):
        targets = []
        metas   = []
        for ann in annotations:
            if len(ann['junctions'])==0: #没有顶点，初始化空tensor
                jmap = torch.zeros([1,self.target_h,self.target_w],device=ann['junctions'].device)
                joff = torch.zeros([2,self.target_h,self.target_w],device=ann['junctions'].device)
                t = {
                    'jloc': jmap.long(),
                    'joff': joff,
                }
                m = {
                    'junc': ann['junctions'],
                    'junc_index': ann['juncs_index'],
                }
            else:
                t,m = self._process_per_image(ann)
            targets.append(t)
            metas.append(m)
        
        return default_collate(targets),metas

    def _process_per_image(self, ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']
        junc_tag = ann['juncs_tag']
        jmap = torch.zeros((height, width), device=device, dtype=torch.int64)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)

        xint, yint = junctions[:,0].long(), junctions[:,1].long()
        off_x = junctions[:,0] - xint.float()-0.5
        off_y = junctions[:,1] - yint.float()-0.5 
        jmap[yint, xint] = junc_tag 
        joff[0, yint, xint] = off_x 
        joff[1, yint, xint] = off_y 
        meta = {
            'junc': junctions,
            'junc_index': ann['juncs_index'],
        }

        target = {
            'jloc': jmap[None],
            'joff': joff,
        }
        return target, meta