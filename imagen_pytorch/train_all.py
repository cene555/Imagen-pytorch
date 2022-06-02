from PIL import Image
import torch as th
from imagen_pytorch.download import load_checkpoint
from imagen_pytorch.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
import argparse
from imagen_pytorch.resample import create_named_schedule_sampler
import os

from imagen_pytorch import logger
from imagen_pytorch.dataset import get_loader
from imagen_pytorch.train_utils import TrainLoop
from imagen_pytorch.get_webdataset_loader import WebdatasetReader

def _fix_path(path):
  d = th.load(path, map_location='cpu')
  checkpoint = {}
  for key in d.keys():
    if not 't5' in key:
        checkpoint[key.replace('module.','')] = d[key]
  return checkpoint

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_folder', type=str, default='', help='Input folder')
  parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
  parser.add_argument('--save_dir', type=str, default='', help='path_for_chaeckpoints')
  parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
  parser.add_argument('--save_interval', type=int, default=200, help='batch_size')
  parser.add_argument('--model_name', type=str, default='t5-3b', help='model_name')
  parser.add_argument('--num_training_steps', type=int, default=2500000, help='num_training_steps')
  parser.add_argument('--num_warmup_steps', type=int, default=10000, help='num_warmup_steps')

  args = parser.parse_args()
  print('num cuda', th.cuda.device_count())
  
  options = model_and_diffusion_defaults()
  options['use_fp16'] = False
  options['t5_name'] = args.model_name
  options['cache_text_emb'] = False
  options['cache_text_emb'] = True
  model, diffusion = create_model_and_diffusion(**options)
  print('start loading')
  if args.checkpoint != '':
    model.load_state_dict(_fix_path(args.checkpoint), strict=False)
  reader = WebdatasetReader(
        args.input_folder,
        args.batch_size,
        2,
        enable_text=True,
        enable_image=True,
        enable_metadata=True,
        t5_name=args.model_name
    )
  data = reader.get_loader()
  logger.configure()

  logger.log("creating model and diffusion...")
  TrainLoop(
        model=model,
        diffusion=diffusion,
        data=reader,
        batch_size=args.batch_size,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=args.save_interval,
        resume_checkpoint=False,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.01,
        lr_anneal_steps=0,
        save_dir=args.save_dir,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps
  ).run_loop()
if __name__ == '__main__':
    main()
