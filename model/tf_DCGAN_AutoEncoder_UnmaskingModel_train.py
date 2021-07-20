from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tqdm.auto import tqdm

import tensorflow as tf

import os, sys
import argparse

def create_maskPair_dataset(
    img_size,
    mask_img_path,
    unmask_img_path,
    train_ratio=0.8,
):
    mask_data = tf.keras.preprocessing.image_dataset_from_directory(mask_img_path, image_size=(img_size, img_size), label_mode=None, shuffle=False)
    unmask_data = tf.keras.preprocessing.image_dataset_from_directory(mask_img_path, image_size=(img_size, img_size), label_mode=None, shuffle=False)
    dataset_length = len(mask_data.file_paths)

    full_dataset = tf.data.Dataset.zip((mask_data.unbatch(), unmask_data.unbatch()))#.shuffle(dataset_length)
    train_dataset = full_dataset.take(int(dataset_length * train_ratio))
    val_dataset = full_dataset.skip(int(dataset_length * train_ratio))

    return train_dataset, val_dataset

def create_unmasking_model(lr=1e-4):
    input = layers.Input(shape=(128, 128, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation="relu", padding="same"
    )(x)
    x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Autoencoder
    autoEncoder = Model(input, x)
    autoEncoder.compile(optimizer=optimizer, loss="mean_squared_error")
    autoEncoder.summary()

    return autoEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_img_path",
        type=str,
        default="../data/celeba-mask-pair/mask_images",
        help="path of mask image",
    )
    parser.add_argument("--mask_img_format", type=str, default="png")
    parser.add_argument(
        "--unmask_img_path",
        type=str,
        default="../data/celeba-mask-pair/unmask_images",
        help="path of unmask image",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="ratio of train data"
    )
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_dataset, val_dataset = create_maskPair_dataset(
        img_size=args.img_size,
        mask_img_path=args.mask_img_path, 
        unmask_img_path=args.unmask_img_path, 
    )
    train_dataset = train_dataset.batch(args.batch_size)
    val_dataset = val_dataset.batch(args.batch_size)

    model = create_unmasking_model(lr=args.lr)

    # reggister callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq=10)
    callback_list = [tensorboard_callback]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callback_list,
    )
    
