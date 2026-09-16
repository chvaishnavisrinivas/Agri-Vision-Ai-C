from datasets import load_dataset

print("Loading PlantVillage dataset...")

dataset = load_dataset(
    "mohanty/PlantVillage",
    "default"
)

print("Dataset loaded!")

print("\nFirst image path:")
print(dataset["train"][0]["text"])

print("\nTrying to access image data...")

image = dataset["train"][0]

print(image)