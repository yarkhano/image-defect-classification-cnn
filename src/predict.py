import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

model = load_model("models/defect_classifier.h5")

def predict_image(model, img_path):
    img = kimage.load_img(img_path, target_size=(64, 64))
    img_array = kimage.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prob = model.predict(img_array)[0][0]
    label = "defective" if prob > 0.5 else "non_defective"
    return label, prob

if __name__ == "__main__":
    test_images = []
    for cls in ["defective", "non_defective"]:
        folder = f"data/processed/test/{cls}"
        first_file = os.listdir(folder)[0]
        test_images.append(os.path.join(folder, first_file))

    for path in test_images:
        label, prob = predict_image(model, path)
        print(f"{path} -> {label} (confidence: {prob:.3f})")