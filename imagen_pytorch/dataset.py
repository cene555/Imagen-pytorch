from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import json
import os
from transformers import AutoTokenizer

        
        
def get_loader(batch_size, resolution, image_dir, df, zero_text_prob=0.1, tokenizer_name='t5-large', max_len=128, shuffle=True,):
    dataset = ImageDataset(resolution, image_dir, df, tokenizer_name, max_len, zero_text_prob)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    return loader
   
class ImageDataset(Dataset):
    def __init__(self, resolution, image_dir, df, tokenizer_name, max_len, zero_text_prob):
        super().__init__()
        self.resolution = resolution
        self.image_dir = image_dir
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.zero_text_prob = zero_text_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        out_dict = {}
        path, text = self.df.iloc[idx]['path'], self.df.iloc[idx]['text']
        
        with bf.BlobFile(os.path.join(self.image_dir, path), "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()


        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y: crop_y + self.resolution, crop_x: crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1
        if np.random.binomial(1, self.zero_text_prob):
            text = ''
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt")

        out_dict["tokens"] = text_encoding['input_ids'][0]
        out_dict["mask"] = text_encoding['attention_mask'][0]
        return np.transpose(arr, [2, 0, 1]), out_dict