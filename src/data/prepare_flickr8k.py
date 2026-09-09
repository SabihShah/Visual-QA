"""
Converts Flickr8k raw files into train/val JSON manifests:
[{"image_path": ..., "caption": ...}, ...]

Expected raw layout (after downloading Flickr8k):
data/flickr8k/Images/*.jpg
data/flickr8k/captions.txt   (image,caption columns, comma or tab separated)
"""
import csv
import json
import os
import random

RAW_DIR = "data/flickr8k"
OUT_DIR = "data"


def main():
    captions_file = os.path.join(RAW_DIR, "captions.txt")
    images_dir = os.path.join(RAW_DIR, "Images")

    samples = []
    with open(captions_file) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 2:
                continue
            image_name, caption = row[0], ",".join(row[1:])
            image_path = os.path.join(images_dir, image_name)
            if os.path.exists(image_path):
                samples.append({"image_path": image_path, "caption": caption.strip()})

    random.seed(42)
    random.shuffle(samples)
    split = int(0.9 * len(samples))
    train, val = samples[:split], samples[split:]

    with open(os.path.join(OUT_DIR, "captioning_train.json"), "w") as f:
        json.dump(train, f)
    with open(os.path.join(OUT_DIR, "captioning_val.json"), "w") as f:
        json.dump(val, f)

    print(f"train: {len(train)} val: {len(val)}")


if __name__ == "__main__":
    main()