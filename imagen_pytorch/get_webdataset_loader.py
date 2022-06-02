from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch

import os
import argparse
import io
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
  
def create_webdataset(
    urls,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    t5_name='t5-11b'
    
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import webdataset as wds  # pylint: disable=import-outside-toplevel


    dataset = wds.WebDataset(wds.ResampledShards(urls))
    print('dataset_created')
    tokenizer_t = AutoTokenizer.from_pretrained(t5_name)
    def tokenizer(text):
        out_dict = {}
        if np.random.binomial(1, 0.08):
            text = ''
        text_encoding = tokenizer_t(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        out_dict["tokens"] = text_encoding['input_ids'][0]
        out_dict["mask"] = text_encoding['attention_mask'][0]
        return out_dict
    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)
    print('dataset filtered')
    resolution = 64
    print('resolution is', resolution)
    def preprocess_dataset(item):
        if enable_image:
            image_data = item[image_key]
            
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image.load()
            while min(*pil_image.size) >= 2 * resolution:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
                )

            scale = resolution / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image.convert("RGB"))
            crop_y = (arr.shape[0] - resolution) // 2
            crop_x = (arr.shape[1] - resolution) // 2
            
            arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
            arr = arr.astype(np.float32) / 127.5 - 1
            
        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)
        return np.transpose(arr, [2, 0, 1]), tokenized_text

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    print('dataset transformed')
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
    )
    return data


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        cache_path=None,
        t5_name='t5-11b',
        
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
            t5_name=t5_name
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")
    def get_loader(self):
        return self.dataloader
    def __iter__(self):
        for batch in self.dataloader:
            yield batch
