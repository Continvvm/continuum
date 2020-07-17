import matplotlib.pyplot as plt
import numpy as np
import os
import imageio


def plot_samples(dataset, title="", path=None, nb_samples=100, shape=None):
    batch, _, _ = dataset.rand_samples(nb_samples)
    filename = os.path.join(path, title)

    if shape is None:
        shape = batch[0].shape

    visualize_batch(batch, nb_samples, shape, filename)


def visualize_batch(batch, number, shape, path):
    batch = batch.cpu().data

    image_frame_dim = int(np.floor(np.sqrt(number)))

    if shape[2] == 1:
        data_np = batch.numpy().reshape(number, shape[0], shape[1], shape[2])
        save_images(data_np[:image_frame_dim * image_frame_dim, :, :, :],
                    [image_frame_dim, image_frame_dim],
                    path)
    elif shape[2] == 3:
        data = batch.numpy().reshape(number, shape[2], shape[1], shape[0])
        make_samples_batche(data[:number], number, path)
    else:
        save_images(batch[:image_frame_dim * image_frame_dim, :, :, :],
                    [image_frame_dim, image_frame_dim],
                    path)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    image -= np.min(image)
    image /= np.max(image) + 1e-12
    image = 255 * image  # Now scale by 255
    image = image.astype(np.uint8)
    return imageio.imwrite(path, image)


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
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

    return img


def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img) + 1e-12
    return img


def make_samples_batche(prediction, batch_size, filename_dest):
    plt.figure()
    batch_size_sqrt = int(np.sqrt(batch_size))
    input_channel = prediction[0].shape[0]
    input_dim = prediction[0].shape[1]
    prediction = np.clip(prediction, 0, 1)
    pred = np.rollaxis(prediction.reshape((batch_size_sqrt,
                                           batch_size_sqrt,
                                           input_channel,
                                           input_dim, input_dim)),
                       2,
                       5)
    pred = pred.swapaxes(2, 1)
    pred = pred.reshape((batch_size_sqrt * input_dim, batch_size_sqrt * input_dim, input_channel))
    fig, ax = plt.subplots(figsize=(batch_size_sqrt, batch_size_sqrt))
    ax.axis('off')
    ax.imshow(img_stretch(pred), interpolation='nearest')
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(filename_dest, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.close()
