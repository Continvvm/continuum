import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(dataset, title="", path=None, nb_samples=100, shape=None, data_type="image_array"):
    batch, y, _ = dataset.get_random_samples(nb_samples)

    if len(y.shape) == 1:
        y, order = y.sort()
        batch = batch[order]

    if path is not None:
        filename = os.path.join(path, title)
    else:
        filename = None

    if shape is None:
        shape = batch[0].shape

    if data_type == "segmentation":
        visualize_segmentation_batch(batch, y, nb_samples, shape, filename)
    else:
        visualize_batch(batch, nb_samples, shape, filename)


def visualize_batch(batch, number, shape, path):
    batch = batch.cpu().data

    image_frame_dim = int(np.floor(np.sqrt(number)))

    if shape[2] == 1:
        data_np = batch.numpy().reshape(number, shape[0], shape[1], shape[2])
        save_images(
            data_np[:image_frame_dim * image_frame_dim, :, :, :],
            [image_frame_dim, image_frame_dim], path
        )
    elif shape[2] == 3:
        data = batch.numpy().reshape(number, shape[2], shape[1], shape[0])
        data = img_stretch(data)
        make_samples_batch(data[:number], number, path)
    else:
        save_images(
            batch[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
            path
        )


def visualize_segmentation_batch(images, segmaps, number, shape, path):
    images, segmaps = images.numpy(), segmaps.numpy()
    images = img_stretch(images)
    make_samples_segmentation_batch(images[:number], segmaps[:number], number, path)


def save_images(images, size, image_path):
    images = np.array(images)
    if images.shape[1] in (1, 3):  # When channel axis is before spatial axis.
        images = images.transpose(0, 2, 3, 1)

    image = np.squeeze(merge(images, size))
    image -= np.min(image)
    image /= np.max(image) + 1e-12
    image = 255 * image  # Now scale by 255
    image = image.astype(np.uint8)

    if image_path is not None:
        return Image.fromarray(image).save(image_path)
    else:
        cmap = None
        if len(image.shape) == 2:
            cmap = "gray"
        plt.axis('off')
        plt.imshow(image, cmap=cmap)


def merge(images, size):
    img = None
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
    else:
        raise ValueError(
            'in merge(images,size) images parameter '
            'must have dimensions: BxHxW or BxHxWx3 or BxHxWx4 '
            f'not {images.shape}'
        )

    return img


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def make_samples_batch(images, batch_size, path):
    plt.figure()
    images, batch_size = _make_square_group(images, batch_size)
    fig, ax = plt.subplots(figsize=(batch_size, batch_size))
    ax.axis('off')
    ax.imshow(images, interpolation='nearest')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])

    if path is not None:
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    plt.close(fig)
    plt.close()


def make_samples_segmentation_batch(images, labels, batch_size, path):
    images, _ = _make_square_group(images, batch_size)
    labels, _ = _make_square_group(labels, batch_size)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].axis('off')
    axes[0].imshow(images, interpolation='nearest')
    axes[0].grid()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # TODO: Add color map for other dataset than VOC
    nclasses = 21
    row_size = images.shape[-1]
    col_size = images.shape[-2]
    cmap = color_map()[:, np.newaxis, :]
    labels = labels[:, :, np.newaxis]
    new_im = np.dot(labels == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(labels == i, cmap[i])
    labels = new_im

    axes[1].axis('off')
    axes[1].imshow(labels, interpolation='nearest')
    axes[1].grid()
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if path is not None:
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    plt.close(fig)
    plt.close()


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap



def _make_square_group(images, batch_size):
    nb_dim = len(images.shape)
    if nb_dim == 3:
        images = np.repeat(images[:, None], 3, axis=1)

    batch_size_sqrt = int(np.sqrt(batch_size))
    input_channel = images[0].shape[0]
    input_dim = images[0].shape[1]

    if nb_dim != 3:
        images = np.clip(images, 0, 1)
    images = images[:batch_size_sqrt ** 2]

    images = np.rollaxis(
        images.reshape((batch_size_sqrt, batch_size_sqrt, input_channel, input_dim, input_dim)),
        2, 5
    )
    images = images.swapaxes(2, 1)
    images = images.reshape((batch_size_sqrt * input_dim, batch_size_sqrt * input_dim, input_channel))

    if nb_dim == 3:
        images = images[..., 0]

    return images, batch_size_sqrt


