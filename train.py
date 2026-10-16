"""
Plant Disease Detector - training script (MobileNetV2 transfer learning)

Dataset: PlantVillage (Kaggle). Folder layout must be:
    data/
      Tomato___Late_blight/ img1.jpg ...
      Tomato___healthy/     img1.jpg ...
      ...
Run:  python train.py
"""
import json
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "data"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_HEAD = 5      # train only the new top layers
EPOCHS_FINE = 5      # then fine-tune the last part of MobileNetV2
SEED = 42

# 1) Load images (80% train / 20% validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("Classes:", len(class_names))
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 2) Augmentation (only active during training)
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 3) Model: MobileNetV2 base + new classifier head
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# 4) Fine-tune: unfreeze the last 30 layers with a small learning rate
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# 5) Save
model.save("plant_model.keras")
print("Saved plant_model.keras and class_names.json")
