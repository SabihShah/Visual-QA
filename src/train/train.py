import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.models.model import VisualCaptioner
from src.data.dataset import CaptioningDataset, VQADataset


def load_samples(json_path):
    import json
    with open(json_path) as f:
        return json.load(f)


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(cfg["text_decoder"])

    model = VisualCaptioner(
        vision_model_name=cfg["vision_encoder"],
        gpt_name=cfg["text_decoder"],
        freeze_vision=cfg["train"]["freeze_vision"],
    ).to(device)

    task = cfg["train"]["task"]
    train_samples = load_samples(f"{cfg['data']['data_dir']}/{task}_train.json")

    if task == "vqa":
        dataset = VQADataset(train_samples, tokenizer,
                              max_q_len=cfg["max_question_len"], max_a_len=cfg["max_answer_len"])
    else:
        dataset = CaptioningDataset(train_samples, tokenizer, max_len=cfg["max_caption_len"])

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]),
    )

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        total_loss = 0
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: avg loss {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/{task}_epoch{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)