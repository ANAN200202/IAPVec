# IAPVec
This is an official implementation of Towards Individual Agricultural Parcel Vectorization from VHR Imagery: A Coarse-to-Fine and Multi-task learning method. 

# Installation
### Environment
Environment reference：[install](https://github.com/KyanChen/RSPrompter#installation)

### SAM pretrain weight
The model is based on Segment Anything Model (SAM), and the pretrain weight of SAM should be placed in pretrained/sam-base.

### Parameters setting
Some parameters in config file need to be modified according to the actual environment.

`work_dir`: The output path of model training.

`hf_sam_pretrain_name`: The name of the SAM model on HuggingFace Spaces. (e.g. IAPVec/pretrained/sam-base)

`hf_sam_pretrain_ckpt_path`: The checkpoint path of the SAM model on HuggingFace Spaces. (e.g. IAPVec/pretrained/sam-base/pytorch_model.bin)

`dataset_type`: The type of dataset, you may modify it according to your own dataset.

`code_root`: The absolute path of the project.

`data_root`: The absolute path of the dataset.

# Model training
### Single card training
```python
python tools/train.py configs/iapvec/iapvec_cascade-capcad-peft-512.py
```

### Multi-card training
```python
bash ./tools/dist_train.sh configs/rsprompter/iapvec_cascade-capcad-peft-512.py ${GPU_NUM}
```

# Model testing
### Single card testing
```python
python tools/test.py configs/rsprompter/iapvec_cascade-capcad-peft-512.py ${CHECKPOINT_FILE}
```

### Multi-card testing
```python
bash ./tools/dist_test.sh configs/rsprompter/iapvec_cascade-capcad-peft-512.py ${CHECKPOINT_FILE} ${GPU_NUM}
```

### Postprocessing
Set `post_process` in config file to **True** when using postprocessing. 

# Acknowledgement
This project is developed based on the [RSPrompter](https://github.com/KyanChen/RSPrompter) and [MMDetection](https://github.com/open-mmlab/mmdetection). Thanks to the developers of these projects.
