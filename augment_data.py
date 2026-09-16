import cv2
import os
import random


input_folders = [
    "dataset/healthy",
    "dataset/stressed"
]

output_folders = [
    "dataset/augmented/healthy",
    "dataset/augmented/stressed"
]


for folder in output_folders:
    os.makedirs(folder, exist_ok=True)


for input_folder, output_folder in zip(input_folders, output_folders):

    for file in os.listdir(input_folder):

        image_path = os.path.join(input_folder, file)

        image = cv2.imread(image_path)

        if image is None:
            continue

        name = os.path.splitext(file)[0]

        # Original image
        cv2.imwrite(
            os.path.join(output_folder, name + "_original.jpg"),
            image
        )

        # Flipped image
        flipped = cv2.flip(image, 1)

        cv2.imwrite(
            os.path.join(output_folder, name + "_flip.jpg"),
            flipped
        )

        # Rotated image
        height, width = image.shape[:2]

        angle = random.choice([-15, 15])

        center = (width // 2, height // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height)
        )

        cv2.imwrite(
            os.path.join(output_folder, name + "_rotate.jpg"),
            rotated
        )


print("Data augmentation completed!")
print("Augmented images saved successfully.")