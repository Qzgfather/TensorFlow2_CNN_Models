import cv2
import numpy as np
from PIL import Image


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            img_path = img_dir + lists.iloc[i, 0] + ".jpg"

            original_img = Image.open(img_path)
            original_img = original_img.resize((dims[0], dims[0]))
            array_img = np.array(original_img) / 255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + ".png")
            resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            array_mask = category_label(resized_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels


if __name__ == '__main__':
    from train import argparser
    import pandas as pd

    args = argparser()
    train_list = pd.read_csv(args.train_list, header=None)
    val_list = pd.read_csv(args.val_list, header=None)
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir
    train_gen = data_gen_small(
        trainimg_dir,
        trainmsk_dir,
        train_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels,
    )

    for i in train_gen:
        print(i)
        break
