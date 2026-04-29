# IAPVec
This is an official implementation of Towards Individual Agricultural Parcel Vectorization from VHR Imagery: A Coarse-to-Fine and Multi-task learning method. 

# Installation
Environment reference：https://github.com/KyanChen/RSPrompter#installation

# Model training
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
