from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class CaptioningDataset(Dataset):
    """Expects a list of {"image_path": str, "caption": str}"""

    def __init__(self, samples, tokenizer, clip_processor_name="openai/clip-vit-base-patch32", max_len=40):
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = CLIPImageProcessor.from_pretrained(clip_processor_name)
        self.max_len = max_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        enc = self.tokenizer(
            item["caption"] + self.tokenizer.eos_token,
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class VQADataset(Dataset):
    """Expects a list of {"image_path": str, "question": str, "answer": str}.
    Question is prepended as context; answer is the target sequence."""

    def __init__(self, samples, tokenizer, clip_processor_name="openai/clip-vit-base-patch32",
                 max_q_len=20, max_a_len=10):
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = CLIPImageProcessor.from_pretrained(clip_processor_name)
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]

        text = f"Question: {item['question']} Answer: {item['answer']}" + self.tokenizer.eos_token
        enc = self.tokenizer(
            text, truncation=True,
            max_length=self.max_q_len + self.max_a_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }