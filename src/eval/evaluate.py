import json
import torch
from transformers import GPT2Tokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from src.models.model import VisualCaptioner
from src.data.dataset import CaptioningDataset


def evaluate(checkpoint, val_json, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(cfg["text_decoder"])

    model = VisualCaptioner(cfg["vision_encoder"], cfg["text_decoder"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    with open(val_json) as f:
        samples = json.load(f)

    gts, res = {}, {}
    for i, item in enumerate(samples):
        from PIL import Image
        from transformers import CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained(cfg["vision_encoder"])
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        pred = model.generate(pixel_values, tokenizer, max_len=cfg["max_caption_len"])[0]
        gts[i] = [item["caption"]]
        res[i] = [pred]

    bleu_score, _ = Bleu(4).compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)

    print(f"BLEU-1..4: {bleu_score}")
    print(f"CIDEr: {cider_score}")


if __name__ == "__main__":
    import yaml
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate("checkpoints/captioning_epoch4.pt", "data/captioning_val.json", cfg)