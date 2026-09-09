import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class PretrainedVQACaptioner:
    def __init__(self, device="cuda", model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens=60) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def caption(self, image: Image.Image) -> str:
        return self._generate(image, "Describe this image in one concise sentence.")

    def answer(self, image: Image.Image, question: str) -> str:
        return self._generate(image, question, max_new_tokens=20)


if __name__ == "__main__":
    model = PretrainedVQACaptioner()
    img = Image.open("data/sample.jpg").convert("RGB")
    print("Caption:", model.caption(img))
    print("Answer:", model.answer(img, "What color is the object?"))