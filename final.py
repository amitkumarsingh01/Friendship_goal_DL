"""FriendshipGoals – Age Group Classification

This script trains a convolutional neural-network with transfer-learning to
classify images into three age-group categories:
    • Adults
    • Teenagers
    • Toddler

It expects the following directory structure (default paths shown):

workspace/
├── train/
│   ├── Adults/
│   ├── Teenagers/
│   └── Toddler/
└── test/
    ├── ImgXXXX.jpg
    └── ...

After training, predictions for the test set are written to the CSV file passed
with --output_csv (default: "submission.csv") in the format required by the
competition (columns: Filename, Category).

Usage
-----
python final.py                             # run with default settings
python final.py --epochs 20 --batch_size 16  # customise training
python final.py --train_dir /path/to/train --test_dir /path/to/test \
              --output_csv my_preds.csv     # custom paths

The script will automatically make use of a GPU if available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def configure_gpus() -> None:
    """Configures TensorFlow to use memory-growth on all available GPUs."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[WARNING] Could not set memory growth on GPU {gpu}: {e}")


def build_model(img_size: tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Builds a classifier based on EfficientNetV2B0 backbone."""

    base_model = efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False  # freeze for initial training

    inputs = layers.Input(shape=(*img_size, 3))
    x = efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EfficientNetV2B0_age_classifier")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train age-group classifier and create submission CSV.")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to training directory with class sub-folders.")
    parser.add_argument("--test_dir", type=str, default="test", help="Path to test images directory.")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Filename for submission CSV.")
    parser.add_argument("--img_size", type=int, default=224, help="Square image size for model input.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=15, help="Total number of epochs (initial + fine-tuning).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of training data used for validation.")
    parser.add_argument("--fine_tune_at", type=int, default=200, help="Unfreeze layers from this index for fine-tuning. Set 0 to train all layers.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

def make_generators(train_dir: str | Path, img_size: tuple[int, int], batch_size: int, val_split: float):
    seed = 123

    train_datagen = ImageDataGenerator(
        preprocessing_function=efficientnet_v2.preprocess_input,
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=efficientnet_v2.preprocess_input,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        subset="training",
        seed=seed,
    )

    val_gen = valid_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        subset="validation",
        seed=seed,
    )

    return train_gen, val_gen


def load_test_images(test_dir: str | Path, img_size: tuple[int, int]) -> tuple[np.ndarray, List[str]]:
    """Loads all images from *test_dir* into a NumPy array and returns them with their filenames."""
    image_paths = sorted(Path(test_dir).glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {test_dir}")

    imgs = []
    filenames = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        imgs.append(img_array)
        filenames.append(img_path.name)

    imgs = np.array(imgs, dtype="float32")
    imgs = efficientnet_v2.preprocess_input(imgs)
    return imgs, filenames


# -----------------------------------------------------------------------------
# Training & inference pipeline
# -----------------------------------------------------------------------------

def train_and_predict(cfg: argparse.Namespace) -> None:
    configure_gpus()

    img_size = (cfg.img_size, cfg.img_size)

    # Data generators
    train_gen, val_gen = make_generators(cfg.train_dir, img_size, cfg.batch_size, cfg.val_split)

    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices  # mapping class_name -> index
    # Inverse mapping index -> class_name to convert predictions back
    index_to_class = {v: k for k, v in class_indices.items()}

    # Build model
    model = build_model(img_size, num_classes)

    # Callbacks
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3, min_lr=1e-6),
        callbacks.ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True),
    ]

    # Phase 1: train classifier head
    steps_per_epoch = max(1, train_gen.samples // cfg.batch_size)
    val_steps = max(1, val_gen.samples // cfg.batch_size)

    print("[INFO] Training classification head ...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(5, cfg.epochs),
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=cbs,
        verbose=1,
    )

    # Phase 2: fine-tune (optional)
    if cfg.epochs > 5:
        print("[INFO] Fine-tuning backbone ...")
        # Unfreeze from fine_tune_at to end
        for layer in model.layers[1].layers[cfg.fine_tune_at:]:
            layer.trainable = True
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.epochs,
            initial_epoch=model.history.epoch[-1] + 1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=cbs,
            verbose=1,
        )

    # ------------------------------------------------------------------
    # Inference on the test set
    # ------------------------------------------------------------------
    print("[INFO] Generating predictions on test images ...")
    test_imgs, test_filenames = load_test_images(cfg.test_dir, img_size)
    preds = model.predict(test_imgs, batch_size=cfg.batch_size, verbose=1)
    pred_indices = np.argmax(preds, axis=1)
    pred_classes = [index_to_class[idx] for idx in pred_indices]

    # Build submission DataFrame
    submission_df = pd.DataFrame({
        "Filename": test_filenames,
        "Category": pred_classes,
    })

    # Ensure the filenames are in the same order as in Test.csv if it exists
    test_csv_path = Path("Test.csv")
    if test_csv_path.exists():
        test_order = pd.read_csv(test_csv_path)
        submission_df = test_order[["Filename"]].merge(submission_df, on="Filename", how="left")

    submission_df.to_csv(cfg.output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"[INFO] Submission file written to {cfg.output_csv} with {len(submission_df)} rows.")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main():
    cfg = parse_args()

    start_time = datetime.now()
    train_and_predict(cfg)
    elapsed = datetime.now() - start_time
    print(f"[INFO] Finished in {elapsed}.")


if __name__ == "__main__":
    # Suppress excessive TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
