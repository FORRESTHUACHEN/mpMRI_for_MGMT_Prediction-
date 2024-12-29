# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:35:16 2024

@author: chenj
"""
import av
import numpy as np
import torch
import os
from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download

np.random.seed(0)

import random

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    if seg_len>800:
        ran_list = random.sample(list(range(clip_len)),clip_len)
        indices = np.array(range(32),dtype=np.int64)
        count=0
        for avl_index in range(clip_len):
            if avl_index in ran_list[:32]:
                indices[count]= avl_index
                count=count+1
        indices=indices*25+2
        # converted_len = int(clip_len * frame_sample_rate)
        # end_idx = np.random.randint(converted_len, seg_len)
        # start_idx = end_idx - converted_len
        # indices = np.linspace(start_idx, end_idx, num=clip_len)
        # indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    else:
        indices = np.array(range(1,seg_len, frame_sample_rate),dtype=np.int64)
        for i in range(32-len(indices)):
            indices = np.append(indices, indices[-1])
    return indices



# video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )



image_processor = VivitImageProcessor.from_pretrained("E:/Datasets/GBM Dataset/Codes/vivitmodel/")
model = VivitForVideoClassification.from_pretrained("E:/Datasets/GBM Dataset/Codes/vivitmodel/")

file_root_path='E:/Datasets/GBM Dataset/MGMTDataset/VideoForFeatures_ver2/'
feature_out_path='E:/Datasets/GBM Dataset/MGMTDataset/DeepFeature_transfor/'
name_dir= os.listdir(file_root_path)
for video_num in range(4000,5004):
    print(name_dir[video_num])
    file_path=file_root_path+name_dir[video_num]
    feature_path=feature_out_path+name_dir[video_num][:-3]+'txt'
    container = av.open(file_path)

    # sample 32 frames

    indices = sample_frame_indices(clip_len=32, frame_sample_rate=25, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    if container.streams.video[0].frames<800:
        for i in range(32-int(container.streams.video[0].frames/25)):
            video = np.append(video,video[-1,:,:,:].reshape(1,256,256,3),axis=0)



    inputs = image_processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    output_features=logits.numpy()
    
    
    f=open(feature_path,'w')

    for f_i in range(400):
        print (output_features[0,f_i],file=f)
    f.close()
