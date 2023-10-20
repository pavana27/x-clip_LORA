from huggingface_hub import hf_hub_download
from ipywidgets import Video
from transformers import XCLIPProcessor, XCLIPModel
import torch


file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
Video.from_file(file_path, width=500)

from decord import VideoReader, cpu
import numpy as np

np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 16 frames
vr.seek(0)
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
video = vr.get_batch(indices).asnumpy()
print(video.shape)

model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

batch_size, num_frames, num_channels, height, width = pixel_values.shape
pixel_values = pixel_values.reshape(-1, num_channels, height, width)

outputs = model(pixel_values)
last_hidden_state = outputs.last_hidden_state
