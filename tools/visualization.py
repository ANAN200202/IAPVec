#coding: utf-8
import sys
sys.path.append('/home/data4/zar24/code/RSPrompter')
import cv2
import mmcv
import numpy as np
import os
import torch
 
from mmdet.apis import inference_detector, init_detector
 
# def featuremap_2_heatmap(feature_map):
#     assert isinstance(feature_map, torch.Tensor)
#     feature_map = feature_map.detach()
#     heatmap = feature_map[:,0,:,:]*0
#     for c in range(feature_map.shape[1]):
#         heatmap+=feature_map[:,c,:,:]
#     heatmap = heatmap.cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
 
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
 
#     return heatmap

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach() # 去掉梯度
    if feature_map.dim() == 4:
        feature_map = feature_map.squeeze(0) # 去掉 batch 维度 [1, C, H, W] -> [C, H, W]
        feature_map = feature_map.mean(dim=0) # 取所有通道的均值 → [H, W]
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        heatmap = feature_map.cpu().numpy()
        # heatmap = cv2.applyColorMap(np.uint8(255 * feature_map.cpu().numpy()), cv2.COLORMAP_JET)
    return heatmap

def draw_feature_map(model, img_path, save_dir):
    '''
    model: 加载参数的模型
    img_path: 测试图像的文件路径
    save_dir: 保存生成图像的文件夹
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = mmcv.imread(img_path)
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector(model, img)  # 这里需要稍微改动检测模型里的simple_test，具体见下面步骤②
    i=2
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.5 + img*0.3  # 这里的0.5是热力图强度因子
        cv2.imwrite(os.path.join(save_dir,'cam_'+'P'+str(i)+'.png'), superimposed_img)  # 保存图像
        i=i+1
 
 
from argparse import ArgumentParser
 
def main():
    parser = ArgumentParser()
    parser.add_argument('--img',default='/home/data4/zar24/data/seven_region/dataset/test/images/003661_sat.jpg', help='Image file') #43,20
    parser.add_argument('--save_dir',default='/home/data4/zar24/code/RSPrompter/heatmap_result/3661_2', help='Dir to save heatmap image')
    parser.add_argument('--config',default='/home/data4/zar24/code/RSPrompter/configs/rsprompter/rsprompter_cascade-seven-peft-512.py', help='Config file')
    parser.add_argument('--checkpoint',
                        # default='work_dirs/rsprompter/rsprompter_cascade-seven/best_coco_segm_mAP_epoch_28.pth', 
                        # default='/home/data4/zar24/code/RSPrompter/work_dirs/rsprompter/rsprompter_cascade-seven-xiaorong1/best_coco_segm_mAP_epoch_21.pth',
                        default='/home/data4/zar24/code/RSPrompter/work_dirs/rsprompter/rsprompter_cascade-seven-no_doundary/best_coco_segm_mAP_epoch_21.pth',
                        # default='work_dirs/rsprompter/rsprompter_cascade-ifly-noedge_offset/best_coco_segm_mAP_epoch_28.pth',
                        # default='work_dirs/rsprompter/rsprompter_cascade-ifly-noedge/best_coco_segm_mAP_epoch_12.pth',
                        # default='work_dirs/rsprompter/nongye_cascade_ifly/best_coco_segm_mAP_epoch_14.pth',
                        help='Checkpoint file')
    parser.add_argument('--device', default='cuda:4', help='Device used for inference')
    args = parser.parse_args()
 
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    draw_feature_map(model,args.img,args.save_dir)
 
if __name__ == '__main__':
    main()