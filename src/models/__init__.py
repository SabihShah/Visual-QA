import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel


class VisionProjector(nn.Module):
    """Projects CLIP image embedding into GPT-2 token embedding space
    as a fixed number of prefix tokens (prefix-tuning style, like ClipCap)."""

    def __init__(self, clip_dim=768, gpt_dim=768, prefix_len=10):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt_dim = gpt_dim
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, gpt_dim * prefix_len // 2),
            nn.GELU(),
            nn.Linear(gpt_dim * prefix_len // 2, gpt_dim * prefix_len),
        )

    def forward(self, clip_embed):
        # clip_embed: (B, clip_dim)
        x = self.proj(clip_embed)
        return x.view(-1, self.prefix_len, self.gpt_dim)


class VisualCaptioner(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32",
                 gpt_name="gpt2", prefix_len=10, freeze_vision=True):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_name)
        gpt_dim = self.gpt.config.n_embd
        clip_dim = self.vision_encoder.config.hidden_size

        self.projector = VisionProjector(clip_dim, gpt_dim, prefix_len)
        self.prefix_len = prefix_len

    def encode_image(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        pooled = out.pooler_output  # (B, clip_dim)
        return self.projector(pooled)  # (B, prefix_len, gpt_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        prefix_embeds = self.encode_image(pixel_values)  # (B, P, D)
        token_embeds = self.gpt.transformer.wte(input_ids)  # (B, T, D)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(prefix_embeds.shape[:2], device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(prefix_embeds.shape[:2], -100, device=labels.device)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None

        return self.gpt(inputs_embeds=inputs_embeds,
                         attention_mask=full_attention_mask,
                         labels=full_labels)

    @torch.no_grad()
    def generate(self, pixel_values, tokenizer, max_len=40, question_ids=None):
        prefix_embeds = self.encode_image(pixel_values)
        if question_ids is not None:
            q_embeds = self.gpt.transformer.wte(question_ids)
            inputs_embeds = torch.cat([prefix_embeds, q_embeds], dim=1)
        else:
            inputs_embeds = prefix_embeds

        generated = self.gpt.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_len,
            do_sample=False,
            num_beams=3,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)