import cv2
import numpy as np
import pandas as pd
from tensorflow.lite.python.interpreter import Interpreter
import os

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./output"


def process_image(model_path, image_path, xlsx_path, min_conf):
    # Load the image
    image_data = cv2.imread(image_path)
    if image_data is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image_data.shape

    # Load the label map
    label_file = ".\\labelmap.txt"
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the nutrient data from the Excel file
    nutrient_data = pd.read_excel(xlsx_path)

    # Load the TFLite model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    # Prepare the input image
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values for floating point models
    if input_details[0]["dtype"] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform inference
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # Get the results
    boxes = interpreter.get_tensor(output_details[1]["index"])[
        0
    ]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]["index"])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]["index"])[0]  # Confidence scores

    # Initialize an array to store detected nutrient information
    detected_nutrients = []

    # Loop over detections and process those with confidence > min_conf
    for i in range(len(scores)):
        if scores[i] > min_conf:
            class_idx = int(classes[i])
            if class_idx >= len(labels):
                print(f"Warning: Detected class index {class_idx} out of range.")
                continue

            object_name = labels[class_idx]  # Get class name

            # Draw bounding box
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))
            cv2.rectangle(image_data, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            label = f"{object_name}: {int(scores[i] * 100)}%"
            cv2.putText(
                image_data,
                label,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # Find the nutrient data for the detected object
            matched_row = nutrient_data[nutrient_data["nama"] == object_name]
            if not matched_row.empty:
                nutrient_info = {
                    "nama": object_name,
                    "karbohidrat": int(matched_row["karbohidrat"].values[0]),
                    "protein": int(matched_row["protein"].values[0]),
                    "lemak": int(matched_row["lemak"].values[0]),
                    "kalori": int(matched_row["kalori"].values[0]),
                    "gula": int(matched_row["gula"].values[0]),
                    "garam": int(matched_row["garam"].values[0]),
                    "air": int(matched_row["air"].values[0]),
                }
                detected_nutrients.append(nutrient_info)

    # Save the output image
    output_image_path = os.path.join(OUTPUT_DIR, "output.jpg")
    cv2.imwrite(output_image_path, image_data)

    return detected_nutrients, output_image_path


# Example usage
# detected_data = process_image_and_extract_nutrients(
#    model=".\\model\\model.tflite",
#    image=".\\sample\\1.jpg",
#    xlsx=".\\data-nutri.xlsx",
#    min_conf=0.3,
# )
# print(detected_data)
