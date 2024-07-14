from omegaconf import OmegaConf
from PIL import Image
import torch

from archs.correspondence_utils import process_image
from extract_hyperfeatures import load_models

import os
import requests

os.environ['http_proxy'] = "http://127.0.0.1:7892"
os.environ['https_proxy'] = "http://127.0.0.1:7892"
# 测试，输出200为正常接通
response = requests.get('http://www.google.com')
print(response.status_code)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else: device = torch.device("cpu")


config, diffusion_extractor, aggregation_network = load_models('./configs/real_one.yaml', device)


""" Extract one image's Hyper-Features """
with torch.inference_mode():
    with torch.autocast("cuda"):
        # Preprocess Image
        path = "./test.png"
        img_pil = Image.open(path).convert("RGB")
        img, _ =  process_image(img_pil, res=(512, 512))  # tensor img
        img = torch.vstack([img.to(device)])
        print(img.size())
        # Extract Unet Layers
        feat, _ = diffusion_extractor.forward(img)
        print("done")
        # Aggregate Features
        diffusion_hyperfeats = aggregation_network(feat.float().view((1, -1, config["output_resolution"], config["output_resolution"])))
        print("done")
        print(diffusion_hyperfeats.size())