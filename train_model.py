import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Parameters
INIT_LR = 1e-4
EPOCHS = 10
BS = 32
IMAGE_SIZE = 224

DATASET_DIR = "dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

print("[INFO] Loading images...")

for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# Load MobileNetV2
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
)

# Build head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training model...")
H = model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    batch_size=BS,
    epochs=EPOCHS
)

# Save model
print("[INFO] Saving model...")
model.save("mask_detector.keras")


print("[SUCCESS] Model training completed and saved!")
