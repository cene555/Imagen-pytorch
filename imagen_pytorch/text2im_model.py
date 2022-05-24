import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .nn import timestep_embedding
from .unet import UNetModel
from .xf import LayerNorm, Transformer, convert_module_to_f16
from transformers import T5EncoderModel


class Text2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        xf_width,
        t5_name,
        *args,
        cache_text_emb=False,
        **kwargs,
    ):
        self.xf_width = xf_width
        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        self.t5 = T5EncoderModel.from_pretrained(t5_name)
        self.t5_proj = nn.Linear(self.t5.shared.embedding_dim, self.model_channels * 4)
        self.to_xf_width = nn.Linear(self.t5.shared.embedding_dim, xf_width)
        self.cache_text_emb = cache_text_emb
        self.cache = None
    def convert_to_fp16(self):

        super().convert_to_fp16()
        self.t5_proj.to(th.float16)
        self.t5.to(th.float16)
        self.to_xf_width.to(th.float16)
    def get_text_emb(self, tokens, mask):
        #with th.no_grad():
        xf_out = self.t5(input_ids=tokens, attention_mask=mask)['last_hidden_state'].detach()
        xf_proj = self.t5_proj(xf_out[:, -1])
        xf_out2 = self.to_xf_width(xf_out)
        xf_out2 = xf_out2.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out2)

        return outputs

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, tokens=None, mask=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out)
        h = h.type(x.dtype)
        h = self.out(h)
        return h
class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

