import os.path as osp
import copy
import einops
import numpy as np
import torch
import cv2
from mmcv.cnn import build_norm_layer, ConvModule
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from peft import get_peft_config, get_peft_model
from torch import nn, Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, \
    SamPromptEncoder, SamVisionEncoderOutput, SamMaskEmbedding
from typing import List, Tuple, Optional
from mmdet.models import CascadeRCNN, CascadeRoIHead, FCNMaskHead, SinePositionalEncoding
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList, DetDataSample
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList
from mmdet.models.losses.focal_loss import FocalLoss
import torch.nn.functional as F

from mmpretrain.models import LayerNorm2d

from math import log
from shapely.geometry import Polygon



@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@MODELS.register_module()
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1 ,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out #bx256x128x128

@MODELS.register_module()
class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        head_size = [[2]]
        heads = []
        for output_channels in sum(head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1) 

@MODELS.register_module()
class IAPVecAnchor(CascadeRCNN): 
    def __init__(
            self,
            shared_image_embedding,
            decoder_freeze=True,
            post_process = False,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.decoder_freeze = decoder_freeze #false
        self.post_process = post_process
        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)

        self.ann_width = 128
        self.ann_height = 128
        from mmdet.iapvec.encoder import Encoder
        self.encoder = Encoder()
        dim_in = 256

        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        self.a2m_att = ECA(dim_in)
        self.a2j_att = ECA(dim_in)
        self.mask_predictor = self._make_predictor(dim_in, 4)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.junc_loss = nn.CrossEntropyLoss()
        self.edge_loss = nn.BCEWithLogitsLoss()
        self.head = MultitaskHead(256, 2)
        self.up = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=4,padding=0)
        self.b_loss = FocalLoss()
        self.mid = nn.ConvTranspose2d(768, 256, kernel_size=4, stride=4,padding=0)
        self.shallow = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),
            LayerNorm2d(256, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        
        self.log_s = torch.nn.Parameter(torch.zeros(3))  

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer
    
    def sigmoid_l1_loss(self, logits, targets, offset = 0.0, mask=None):
        logp = torch.sigmoid(logits) + offset
        loss = torch.abs(logp-targets)

        if mask is not None:
            t = ((mask == 1) | (mask == 2)).float()
            w = t.mean(3, True).mean(2,True)
            w[w==0] = 1
            loss = loss*(t/w)

        return loss.mean()

    def extract_jloc(self, features):
        mask_feature = self.mask_head(features) 
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)
        #ECA
        mask_att_feature = self.a2m_att(afm_feature, mask_feature) 
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature) 
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature) 

        return mask_pred, jloc_pred
    
    def build_juncs(self, results, shape):
        width = shape[0]
        height = shape[1]
        ann = {
            'junctions': [], 
            'juncs_index': [], 
            'juncs_tag': [], 
            'edges_positive': [], 
            'width': width,
            'height': height,
        }

        pid = 0
        instance_id = 0
        for ann_per_ins in results:  
            juncs, tags = [], []
            segmentations = ann_per_ins 
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2) # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4) 
                points = segm 
                junc_tags = np.ones(points.shape[0]) 
                if i == 0:  # outline
                    poly = Polygon(points) 
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords) 
                        convex_index = [(p == convex_point).all(1).any() for p in points] 
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2    
                        tags.extend(junc_tags.tolist())
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    
            idxs = np.arange(len(juncs)) 
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1,0) + pid 

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist()) 
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)  

        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.int64],
                           ['juncs_tag', np.int64],
                           ['juncs_index', np.int64],
                           ):
            ann[key] = np.array(ann[key], dtype=_type) 

        return ann

    def annot_resize(self, ann):

        sx = self.ann_width / ann['width']
        sy = self.ann_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        return ann
    
    def totensor(self, ann, device):
        for key, val in ann.items():
            if isinstance(val, np.ndarray):
                ann[key] = torch.from_numpy(val)
                ann[key] = ann[key].to(device) 
        return ann
    
    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def weighted_cross_entropy_loss(self, preds, edges):
        """ Calculate sum of weighted cross entropy loss. """
        mask = (edges > 0.5).float()   
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos                     # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight.masked_scatter_(edges > 0.5,
            torch.ones_like(edges) * num_neg / (num_pos + num_neg))
        weight.masked_scatter_(edges <= 0.5,
            torch.ones_like(edges) * num_pos / (num_pos + num_neg))

        losses = F.binary_cross_entropy_with_logits(
            preds.float(), edges.float(), weight=weight, reduction='none')
        loss = torch.sum(losses) / b
        return loss
    
    def pointline_multitask(self,x,mid,batch_data_samples,batch_inputs):
        device = batch_inputs.device
        mid_reshape = mid.permute(0, 3, 1, 2)
        mid_reshape = self.mid(mid_reshape)
        x_shallow = self.shallow(mid_reshape+x[0])

        outputs = self.head(x_shallow) 
        juncs_annots = []
        edge_pics = np.zeros((len(batch_data_samples),batch_inputs.shape[2],batch_inputs.shape[3]))
        c = 0
        for data_sample in batch_data_samples:
            shape = data_sample.batch_input_shape 
            mask_junctions = data_sample.gt_instances.masks.masks 
            juncs_annot = self.build_juncs(mask_junctions, shape) 
            if len(juncs_annot['junctions']): 
                juncs_annot = self.annot_resize(juncs_annot) 
            juncs_annot = self.totensor(juncs_annot, device) 
            juncs_annots.append(juncs_annot)
            polys = data_sample.gt_instances.masks.masks
            edge_pic = np.zeros((shape[0],shape[1],3))
            color_poly=(255,255,255)
            if polys:
                for poly in polys:
                    poly = poly[0]
                    poly = poly.reshape(int(poly.shape[0]/2),2)
                    poly = poly.reshape((-1,1,2)).astype(np.int_)
                    cv2.polylines(edge_pic,[poly],True,color_poly,1) 
            edge_pic = edge_pic[:,:,0]/255
            edge_pics[c] = edge_pic
            c = c+1
        edge_pics = torch.from_numpy(edge_pics).to(device)

        edge_pics = edge_pics.to(torch.float32)
        edge_pics = edge_pics.unsqueeze(1) 

        targets, metas = self.encoder(juncs_annots) 

        juncs_losses = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_edge': 0.0
        }

        edge_pred, jloc_pred = self.extract_jloc(x_shallow) 

        edge_pred = self.up(edge_pred) 

        if targets is not None:
            juncs_losses['loss_jloc'] += self.junc_loss(jloc_pred, targets['jloc'].squeeze(dim=1))
            juncs_losses['loss_joff'] += self.sigmoid_l1_loss(outputs[:, :], targets['joff'], -0.5, targets['jloc'])
            juncs_losses['loss_edge'] += self.weighted_cross_entropy_loss(edge_pred, edge_pics)/10000 
        
        return juncs_losses
            
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs, mid = self.backbone(batch_inputs) 
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple): 
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        x = self.neck(vision_hidden_states) 

        return x, image_embeddings, image_positional_embeddings, mid

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:

        x, image_embeddings, image_positional_embeddings, mid = self.extract_feat(batch_inputs) 

        
        losses = dict()
        juncs_losses = self.pointline_multitask(x,mid,batch_data_samples,batch_inputs)
        
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.masks=data_sample.gt_instances.masks.to_bitmap()
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)  
        


        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg) 
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        for data_sample in batch_data_samples:
            data_sample.gt_instances.masks=data_sample.gt_instances.masks.to_bitmap()
            
        roi_losses = self.roi_head.loss(
            x, rpn_results_list, batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        losses.update(juncs_losses)
        losses.update(roi_losses)
        
        return losses
    
    def remove_small_regions(self,
        mask: np.ndarray, area_thresh: float, mode: str
    ) -> Tuple[np.ndarray, bool]:
        """
        Removes small disconnected regions and holes in a mask. Returns the
        mask and an indicator of if the mask has been modified.
        """
        #import cv2  # type: ignore

        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8) 
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 2]
        mask = np.isin(regions, fill_labels)
        return mask
    
    def postprocess(self,results_list,device):
        area_thresh = 30
        for t in range(len(results_list)):
            result_masks = results_list[t].masks.int().cpu().numpy()
            result_masks = result_masks.astype(np.uint8)  #nx512x512,n为mask个数，由0或1组成
            for b in range(result_masks.shape[0]):
                # remove holes &islands
                temp = result_masks[b,:,:]
                temp = self.remove_small_regions(temp, area_thresh, mode="holes")
                temp = temp.astype(np.uint8)
                temp = self.remove_small_regions(temp, area_thresh, mode="islands")
                temp = temp.astype(np.uint8)
                # result_masks[b] = temp
                
                # Douglas-Peuker
                conts = None
                contours, hierarchy = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)==1:
                    for contour in contours:
                        if contour.shape[0]>=4:
                            epsilon = 0.005 * cv2.arcLength(contour, True)                       
                            simplified_polygon = cv2.approxPolyDP(contour, epsilon, True)
                            contour = np.squeeze(contour) #nx2
                            simplified_polygon = np.squeeze(simplified_polygon) #nx2
                            coords_c = contour.tolist()
                            coords_s = simplified_polygon.tolist()
                            if len(coords_s)>=4:
                                new_polygon = Polygon(coords_s)
                                old_polygon = Polygon(coords_c)
                                area1 = new_polygon.area
                                area2 = old_polygon.area
                                if area2>0:
                                    r = area1 / area2
                                    if (r<=1.05) & (r>=0.95):
                                        conts = simplified_polygon
                                    else:
                                        conts = contour
                                else:
                                    conts = contour
                            else:
                                conts = contour
                if conts is None:
                    result_masks[b] = temp
                else:
                    polymask = np.zeros((512,512))
                    conts = conts[:,np.newaxis,:]
                    conts = [conts]
                    cv2.drawContours(polymask, conts, -1, 1, -1)
                    result_masks[b] = polymask.astype(np.uint8)
            results_list[t].masks = torch.from_numpy(result_masks).bool().to(device)
        
        # remove overlapped parcels
        for t in range(len(results_list)):
            result_masks_p = results_list[t].masks.cpu().numpy()
            result_scores = results_list[t].scores.cpu().numpy()
            result_masks = result_masks_p.astype(np.float64)  
            roi_num = result_masks.shape[0]
            if roi_num!=0:
                w = result_masks.shape[1]
                h = result_masks.shape[2]
                result_lines = result_masks.reshape(roi_num,w*h)
                masks_area = np.sum(result_lines,axis=1) 
                masks_area = masks_area[np.newaxis,:]
                masks_area = masks_area.repeat(roi_num,axis=0) 
                result_lines_t = result_lines.transpose(1,0)
                masks_inter = np.matmul(result_lines,result_lines_t) 
                mut_matr = np.ones(roi_num)-np.eye(roi_num)
                masks_inter = masks_inter*mut_matr
                rate = masks_inter/masks_area 
                overlap_roi = np.zeros((1,roi_num),dtype=bool)
                for n in range(roi_num):
                    roi = rate[:,n]
                    overlap_index = roi>=0.90  
                    overlap_rate1 = roi[overlap_index]
                    overlap_rate2 = np.where(overlap_index)[0]
                    for p in range(overlap_rate1.shape[0]):
                        # p1 = overlap_rate1[p]
                        # p2 = rate[n,overlap_rate2[p]]
                        #if p1>p2:
                        if result_scores[n]<result_scores[overlap_rate2[p]]:
                            #if p2>=0.8:
                            overlap_roi[0,n] = True
                            # else:
                            #     overlap_roi[0,overlap_rate2[p]] = True
                # result_scores[overlap_roi[0]] = 0
                result_scores[overlap_roi[0]] = result_scores[overlap_roi[0]]-0.3
                result_scores[np.where(masks_area[0]==0)] = 0
                results_list[t].scores = torch.from_numpy(result_scores).to(device)
        return results_list
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings, mid = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        for data_sample in batch_data_samples:  
            data_sample.gt_instances.masks=data_sample.gt_instances.masks.to_bitmap()
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        #以下是加入开闭运算和道格拉斯
        if self.post_process:
            device = batch_inputs.device
            results_list = self.postprocess(results_list,device)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)  

        return batch_data_samples
    
    def _forward(self,
            batch_inputs: Tensor,
            batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings, mid = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        results = ()
        results = results + (results_list[0].masks, )
        
        return results

@MODELS.register_module()
class RSSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class RSSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            peft_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        vision_encoder = SamVisionEncoder(sam_config)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[(r'^module\.', ''), (r'^vision_encoder\.', '')])

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)

@MODELS.register_module()
class MMPretrainSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            img_size=1024,
            peft_config=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)  
        vision_encoder_cfg = dict(
            type='mmpretrain.ViTSAM',
            arch=hf_pretrain_name.split('-')[-1].split('_')[-1],
            img_size=img_size,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg) 
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),
                map_location='cpu',
                revise_keys=[
                    (r'^module\.', ''),
                    (r'^vision_encoder\.', ''),
                    (r'.layer_norm1.', '.ln1.'),
                    (r'.layer_norm2.', '.ln2.'),
                    (r'.mlp.lin1.', '.ffn.layers.0.0.'),
                    (r'.mlp.lin2.', '.ffn.layers.1.'),
                    (r'neck.conv1.', 'channel_reduction.0.'),
                    (r'neck.ln1.', 'channel_reduction.1.'),
                    (r'neck.conv2.', 'channel_reduction.2.'),
                    (r'neck.ln2.', 'channel_reduction.3.'),
                ]
            )

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)
        self.input_image_size = 512
        self.shared_embedding = self.forward_with_coords
        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, sam_config.hidden_size) for i in range(sam_config.num_point_embeddings)]
        )
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            1.0 * torch.randn((2, 128)),
        )
        self.mask_embed = SamMaskEmbedding(sam_config)
        self.not_a_point_embed = nn.Embedding(1, sam_config.hidden_size)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        #coords = torch.tensor(coords,dtype=torch.float16)
        coords = coords @ self.positional_encoding_gaussian_matrix
        #coords = torch.tensor(coords,dtype=torch.float32)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)


@MODELS.register_module()
class RSFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class PseudoFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.channel_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(hidden_channels, eps=1e-6),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels, eps=1e-6),
        )

    def forward(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = self.channel_fusion(x)
        return x
    
    
@MODELS.register_module()
class RSFeatureAggregator(BaseModule):
    in_channels_dict = {
        'base': [768] * (12+1),
        'large': [1024] * (24+1),
        'huge': [1280] * (32+1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 12, 2),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x

@MODELS.register_module()
class RSSimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]



        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)


@MODELS.register_module()
class IAPVecAnchorRoIPromptHead(CascadeRoIHead): 
    def __init__(
        self,
        with_extra_pe=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if with_extra_pe:
            # out_channels = self.bbox_roi_extractor.out_channels
            out_channels = 256
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,
            )
            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def _mask_forward(self,
                      stage: int,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      image_embeddings=None,
                      image_positional_embeddings=None
                      ) -> dict:
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, iou_predictions = self.mask_head(    
            x,
            mask_feats,
            rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def mask_loss(self, 
                  stage: int,
                  x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList,
                  image_embeddings=None,
                  image_positional_embeddings=None
                  ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                print('no pos rois')
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            mask_results = self._mask_forward(
                stage, x, pos_rois, 
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target,mask_pred = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage]
            )

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             # extra inputs
             image_embeddings=None,
             image_positional_embeddings=None,
             ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)

        losses = dict()

        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]

                for i in range(num_imgs):
                    results = results_list[i]
                    # rename rpn_results.bboxes to rpn_results.priors
                    #if 'bboxes' in results:
                    results.priors = results.pop('bboxes')

                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self.bbox_loss(stage, x, sampling_results)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask and stage==2:
                mask_results = self.mask_loss(
                    stage, x, sampling_results, bbox_results['bbox_feats'], batch_gt_instances,
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_positional_embeddings,
                )
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results=sampling_results,
                        bbox_results=bbox_results,
                        batch_img_metas=batch_img_metas)
                    # Empty proposal
                    if results_list is None:
                        break

        return losses

    def predict_mask(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
        ) -> InstanceList:

        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list
        
        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        mask_results = self._mask_forward(
        2, x, mask_rois,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for i in range(len(batch_img_metas)):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_masks.append(aug_mask[0])
        results_list = self.mask_head.predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        
        return results_list

    #下面是anchor的推理
    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                # extra inputs
                image_embeddings=None,
                image_positional_embeddings=None,
                ) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return results_list


@MODELS.register_module()
class IAPVecAnchorMaskHead(FCNMaskHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_conv_res: bool = True,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None,
            *args,
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='RSSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        self.prompt_encoder = MODELS.build(prompt_encoder)
        self.prompt_encoder.init_weights()
        self.no_mask_embed = self.prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels*roi_feat_size**2//4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * self.per_pointset_point)
        )

        self.loss_mask = MODELS.build(loss_mask)

        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)
    
    def get_targets(self, sampling_results: List[SamplingResult],
                batch_gt_instances: InstanceList,
                rcnn_train_cfg: ConfigDict,
                distance = False) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results] 
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ] 
        device = pos_proposals[0].device
        gt_masks = [res.masks for res in batch_gt_instances]
        gt = torch.zeros((len(gt_masks),512,512), device=device, dtype=torch.float32)
        t=0
        for gt_m in gt_masks:
            temp = torch.from_numpy(gt_m.masks).float().to(device)
            sum_temp = torch.sum(temp,dim=0)
            if sum_temp is not None:
                gt[t]=sum_temp
            t = t+1
        mask_targets_list = []
        if distance:
            distance_maps = []
        mask_size = (512,512)
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if distance:
                distance_results = []
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
                distance_map = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
                if distance:
                    eps = 0.00001
                    for i in range(len(mask_targets)):
                        mask_target = mask_targets[i].cpu().detach().numpy().astype(np.uint8)
                        mask_target[mask_target>0] = 255
                        distance_result = cv2.distanceTransform(src=mask_target, distanceType=cv2.DIST_L2, maskSize=3)
                        min_value = np.min(distance_result)
                        max_value = np.max(distance_result)
                        scaled_image = (distance_result - min_value+eps) / (max_value - min_value+eps)
                        distance_result = scaled_image
                        distance_result = distance_result.astype(np.float32)
                        distance_results.append(distance_result)
                    distance_map = torch.from_numpy(np.stack(distance_results, axis=0)).to(device)
            if distance:
                distance_maps.append(distance_map)
            mask_targets_list.append(mask_targets) 
        mask_targets = torch.cat(mask_targets_list)
        if distance:
            distance_targets = torch.cat(distance_maps)
        
        if distance:
            return mask_targets, gt, device, distance_targets
        else:
            return mask_targets, gt, device
        
    def forward(self,
                multifeature, 
                x,
                rois,
                image_embeddings, 
                image_positional_embeddings, 
                roi_img_ids=None
                ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0] 

        rois = rois[:,1:]
        rois0 = rois[:,0].clone()
        rois1 = rois[:,1].clone()
        rois2 = rois[:,2].clone()
        rois3 = rois[:,3].clone()
        rois[:,0] = rois1
        rois[:,1] = rois0
        rois[:,2] = rois3
        rois[:,3] = rois2
        rois = rois.unsqueeze(0)
        bbox_embeddings = self.prompt_encoder._embed_boxes(rois) 
        bbox_embeddings = bbox_embeddings.transpose(0,1)

        image_embedding_size = image_embeddings.shape[-2:]
        point_embedings = self.point_emb(x) 
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c', n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2] 
        
        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)

        sparse_embeddings = torch.cat([sparse_embeddings, bbox_embeddings], dim=2)

        num_roi_per_image = torch.bincount(roi_img_ids.long())
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image, torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device, dtype=num_roi_per_image.dtype)])

        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0], image_embedding_size[1])

        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings=image_embeddings, 
            image_positional_embeddings=image_positional_embeddings, 
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=self.multimask_output, 
            attention_similarity=self.attention_similarity, 
            target_embedding=self.target_embedding, 
            output_attentions=self.output_attentions, 
        )

        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w) 
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        
        return low_res_masks, iou_predictions
    
    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets, gt, device = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg,
            distance=False) #gt

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False) 

        loss = dict()
        if mask_preds.size(0) == 0: 
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels)) 

            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)

        loss['loss_mask'] = loss_mask 
        return dict(loss_mask=loss, mask_targets=mask_targets),mask_preds

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear', align_corners=False).squeeze(1) #n×512×512

        scale_factor_w, scale_factor_h = img_meta['scale_factor'] #1
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]

        h, w = img_meta['ori_shape'] 
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0: 
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask