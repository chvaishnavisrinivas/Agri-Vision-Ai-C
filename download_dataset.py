from datasets import load_dataset

print("Starting PlantVillage dataset download...")

dataset = load_dataset("mohanty/PlantVillage", "default")

print("Dataset downloaded successfully!")
print(dataset)