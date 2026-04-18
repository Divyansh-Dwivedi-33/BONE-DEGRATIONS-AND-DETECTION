"""
training.py
===========
Knee bone condition classifier training script.

Expected dataset layout:
    MLPROJ/ml dataset knees/
        Normal/
        Osteopenia/
        Osteoporosis/

This script also auto-detects a sibling Desktop folder named
"OS Collected Data" with the same class structure.

Run:
    python training.py

Output:
    knee_model.keras
"""

import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_SAVE_PATH = str(BASE_DIR / "knee_model.keras")
TRAINING_CURVES_PATH = str(BASE_DIR / "training_curves.png")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # remainder

SPLIT_DIR = str(BASE_DIR / "data_split")


def resolve_dataset_path() -> str:
    """Find the dataset folder from common local locations."""
    candidates = [
        BASE_DIR / "MLPROJ" / "ml dataset knees",
        BASE_DIR.parent / "MLPROJ" / "ml dataset knees",
        BASE_DIR / "OS Collected Data",
        BASE_DIR.parent / "OS Collected Data",
    ]

    for candidate in candidates:
        if candidate.is_dir() and all((candidate / cls).is_dir() for cls in CLASSES):
            return str(candidate)

    checked_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the dataset folder. Checked:\n"
        f"{checked_paths}"
    )


def count_split_images(split_dir: str) -> int:
    """Count images already present in the cached train/val/test split."""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    return sum(
        1
        for _, _, files in os.walk(split_dir)
        for filename in files
        if filename.lower().endswith(image_extensions)
    )


def prepare_split_dirs(dataset_path: str, split_dir: str) -> None:
    """
    Create a train/val/test folder structure by copying images from the
    original dataset. Reuses an existing non-empty split.
    """
    existing_images = count_split_images(split_dir) if os.path.exists(split_dir) else 0
    if existing_images > 0:
        print(f"[INFO] Split directory '{split_dir}' already contains {existing_images} images. Skipping split.")
        return

    if os.path.exists(split_dir):
        print(f"[INFO] Split directory '{split_dir}' exists but is empty. Rebuilding split.")
    else:
        print("[INFO] Splitting dataset into train / val / test ...")

    for subset in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(split_dir, subset, cls), exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for cls in CLASSES:
        cls_path = os.path.join(dataset_path, cls)
        images = [
            filename for filename in os.listdir(cls_path)
            if filename.lower().endswith(image_extensions)
        ]

        np.random.seed(SEED)
        np.random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for subset, files in splits.items():
            for filename in files:
                src = os.path.join(cls_path, filename)
                dst = os.path.join(split_dir, subset, cls, filename)
                shutil.copy2(src, dst)

        print(
            f"  {cls}: {len(splits['train'])} train | "
            f"{len(splits['val'])} val | {len(splits['test'])} test"
        )

    print("[INFO] Dataset split complete.\n")


def build_generators(split_dir: str):
    """Return train, validation, and test generators plus class indices."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )

    val_test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_test_aug.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    test_gen = val_test_aug.flow_from_directory(
        os.path.join(split_dir, "test"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen, train_gen.class_indices


def build_model(num_classes: int = 3) -> tf.keras.Model:
    """Build a DenseNet121 transfer-learning classifier."""
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="KneeClassifier")


def train_model(model, train_gen, val_gen) -> tf.keras.callbacks.History:
    """Compile and train the model."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    print("\n[INFO] Starting training ...")
    return model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )


def plot_history(history: tf.keras.callbacks.History) -> None:
    """Save training accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(TRAINING_CURVES_PATH, dpi=150)
    print(f"[INFO] Training curves saved to {TRAINING_CURVES_PATH}")
    plt.show()


def evaluate_model(model, test_gen) -> None:
    """Evaluate the model on the test split."""
    print("\n[INFO] Evaluating on test set ...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")


def main():
    dataset_path = resolve_dataset_path()
    print(f"[INFO] Using dataset at '{dataset_path}'.")

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Trained model already found at '{MODEL_SAVE_PATH}'.")
        print("[INFO] Delete it to retrain. Exiting.")
        return

    prepare_split_dirs(dataset_path, SPLIT_DIR)

    train_gen, val_gen, test_gen, class_indices = build_generators(SPLIT_DIR)
    print(f"\n[INFO] Class indices: {class_indices}")

    model = build_model(num_classes=len(CLASSES))
    model.summary()

    history = train_model(model, train_gen, val_gen)

    plot_history(history)
    evaluate_model(model, test_gen)

    # Save the final restored model in native Keras format.
    model.save(MODEL_SAVE_PATH)

    print(f"\n[SUCCESS] Model saved to {MODEL_SAVE_PATH}")
    print("[INFO] Run app.py to start the prediction interface.")


if __name__ == "__main__":
    main()
