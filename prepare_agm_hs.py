from datasets import load_dataset
import os
import random

print("Loading AGM_HS dataset...")
dataset = load_dataset("deep-plants/AGM_HS")["train"]

healthy = [i for i, x in enumerate(dataset["label"]) if x == "healthy"]
stressed = [i for i, x in enumerate(dataset["label"]) if x == "stressed"]

random.seed(42)
random.shuffle(healthy)
random.shuffle(stressed)

# 800 healthy + 800 stressed for training
# 200 healthy + 200 stressed for testing
train_healthy = healthy[:800]
test_healthy = healthy[800:1000]

train_stressed = stressed[:800]
test_stressed = stressed[800:1000]

folders = [
    "dataset/AGM_HS/healthy",
    "dataset/AGM_HS/stressed",
    "dataset/AGM_HS/test/healthy",
    "dataset/AGM_HS/test/stressed"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Saving training images...")

for count, index in enumerate(train_healthy):
    image = dataset[index]["image"]
    image.save(f"dataset/AGM_HS/healthy/healthy_{count}.jpg")

for count, index in enumerate(train_stressed):
    image = dataset[index]["image"]
    image.save(f"dataset/AGM_HS/stressed/stressed_{count}.jpg")

print("Saving testing images...")

for count, index in enumerate(test_healthy):
    image = dataset[index]["image"]
    image.save(f"dataset/AGM_HS/test/healthy/healthy_test_{count}.jpg")

for count, index in enumerate(test_stressed):
    image = dataset[index]["image"]
    image.save(f"dataset/AGM_HS/test/stressed/stressed_test_{count}.jpg")

print("Dataset preparation completed!")
print("Training: 800 healthy + 800 stressed")
print("Testing: 200 healthy + 200 stressed")