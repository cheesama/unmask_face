from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from tqdm.auto import tqdm

import tensorflow as tf

import os, sys
import multiprocessing
import argparse

def create_maskPair_dataset(
    mask_img_path,
    unmask_img_path,
    mask_prefix="_cloth",
    mask_img_format="png",
    unmask_img_format="jpg",
    train_ratio=0.8,
):
    manager = multiprocessing.Manager()
    image_list = manager.list()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    def add_image_data(maskImg):
        maskImgName = f'{maskImg.split(".")[0]}.{mask_img_format}'
        unmaskImgName = f'{maskImgName.replace(mask_prefix, "")}.{unmask_img_format}'

        if unmaskImgName in os.listdir(unmask_img_path):
            image_pair = []
            image_pair.append(img_to_array(load_img(os.path.join(mask_img_path, maskImgName), target_size=(128, 128))))
            image_pair.append(img_to_array(load_img(os.path.join(unmask_img_path, unmaskImgName), target_size=(128, 128))))
            image_list.append(image_pair)
        else:
            print (f'check image pair info: {maskImgName}')

    print ('loading image data ...')
    pool.map(add_image_data, os.listdir(mask_img_path))
    pool.close()

    full_dataset = tf.data.Dataset.from_tensor_slices(([img_pair[0] for img_pair in image_list], [img_pair[1] for img_pair in image_list])).shuffle(len(image_list))
    train_dataset = full_dataset.take(int(len(image_list) * train_ratio))
    val_dataset = full_dataset.skip(int(len(image_list) * train_ratio))

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
        default="../data/img_align_celeba_png_masked",
        help="path of mask image",
    )
    parser.add_argument("--mask_img_format", type=str, default="png")
    parser.add_argument(
        "--unmask_img_path",
        type=str,
        default="../data/img_align_celeba_png",
        help="path of unmask image",
    )
    parser.add_argument("--unmask_img_format", type=str, default="jpg")
    parser.add_argument(
        "--mask_prefix", type=str, default="_cloth", help="prefix of mask image"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="ratio of train data"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_dataset, val_dataset = create_maskPair_dataset(
        mask_img_path=args.mask_img_path, 
        mask_img_format=args.mask_img_format,
        unmask_img_path=args.unmask_img_path, 
        unmask_img_format=args.unmask_img_format,
        mask_prefix=args.mask_prefix, 
        train_ratio=args.train_ratio
    )
    train_dataset = train_dataset.shuffle(len(train_dataset)).batch(args.batch_size)
    val_dataset = val_dataset.batch(args.batch_size)

    model = create_unmasking_model(lr=args.lr)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
    )
    
