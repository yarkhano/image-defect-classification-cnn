import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (64, 64)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "data/processed/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=["non_defective", "defective"]
)
val_gen = val_test_datagen.flow_from_directory(
    "data/processed/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=["non_defective", "defective"], shuffle=False
)
test_gen = val_test_datagen.flow_from_directory(
    "data/processed/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=["non_defective", "defective"], shuffle=False
)


from tensorflow.keras import layers, models

def build_cnn():

    model = models.Sequential([

        layers.Input(shape=(64, 64, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),

        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),

        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),

        layers.MaxPooling2D(2, 2),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")  # binary output

    ])

    model.compile(

        optimizer="adam",

        loss="binary_crossentropy",

        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]

    )

    return model

model = build_cnn()

model.summary()



history = model.fit(

    train_gen,

    validation_data=val_gen,

    epochs=20,

    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]

)



import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)

print(f"Test Accuracy: {test_acc:.3f} | Precision: {test_prec:.3f} | Recall: {test_rec:.3f}")

y_true = test_gen.classes

y_pred_probs = model.predict(test_gen)

y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(classification_report(y_true, y_pred, target_names=["non_defective", "defective"]))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",

            xticklabels=["non_defective", "defective"],

            yticklabels=["non_defective", "defective"])

plt.title("Confusion Matrix - CNN Defect Classifier")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("reports/confusion_matrix_cnn.png")

plt.show()



plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="train")

plt.plot(history.history["val_accuracy"], label="val")

plt.title("Accuracy")

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="train")

plt.plot(history.history["val_loss"], label="val")

plt.title("Loss")

plt.legend()

plt.savefig("reports/training_curves.png")

plt.show()



def visualize_activation_map(model, img_path):

    from tensorflow.keras.preprocessing import image as kimage

    img = kimage.load_img(img_path, target_size=(64, 64))

    img_array = kimage.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    layer_outputs = [layer.output for layer in model.layers if "conv2d" in layer.name]

    activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

    activations = activation_model.predict(img_array)

    first_layer_activation = activations[0]

    plt.figure(figsize=(10, 8))

    for i in range(8):

        plt.subplot(2, 4, i + 1)

        plt.imshow(first_layer_activation[0, :, :, i], cmap="viridis")

        plt.axis("off")

    plt.suptitle("First Conv Layer Activations")

    plt.savefig("reports/activation_maps.png")

    plt.show()



model.save("models/defect_classifier.h5")

print("Model saved to models/defect_classifier.h5")



def predict_image(model, img_path):

    from tensorflow.keras.preprocessing import image as kimage

    img = kimage.load_img(img_path, target_size=(64, 64))

    img_array = kimage.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]

    label = "defective" if prob > 0.5 else "non_defective"

    return label, prob

label, prob = predict_image(model, "data/processed/test/defective/img_0.png")

print(f"Prediction: {label} ({prob:.3f})")







