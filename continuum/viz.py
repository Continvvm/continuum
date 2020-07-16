import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize


def sample(dataset, nb_per_class=5):
    sampled_x, sampled_y = [], []

    _x, _y, _t = dataset.get_data()

    for class_id in np.unique(_y):
        indexes = np.where(_y == class_id)[0][:nb_per_class]
        x, y = dataset.get_samples_from_ind(indexes)
        sampled_x.append(x.numpy())
        sampled_y.append(y)
    sampled_x = np.concatenate(sampled_x)
    sampled_y = np.concatenate(sampled_y)

    return sampled_x, sampled_y


def plot_old(dataset, title="", path=None, nb_per_class=5, shape=None):
    # x, y = sample(dataset.x, dataset.y, nb_per_class=nb_per_class)
    # if not dataset.open_image and x.shape[1] == 1:
    #    x = x.squeeze(1)

    print("start")

    big_image = None
    cmap = None
    for class_id in range(dataset.nb_classes):
        for sample_id in range(nb_per_class):
            x_, y_ = dataset.__getitem__(class_id * nb_per_class + sample_id)
            print(x_.shape)

            if dataset.open_image:
                img = np.asarray(x_)
            elif x_.shape[0] == 1:  # Grayscale, no channel dimension
                img = x_.squeeze(0)
                cmap = "gray"

            if big_image is None:
                if shape is None:
                    h = img.shape[0]
                    w = img.shape[1]
                else:
                    h, w = shape

                if cmap == "gray":
                    big_image = np.empty(
                        (dataset.nb_classes * h, nb_per_class * w), dtype=img.dtype
                    )
                else:
                    big_image = np.empty(
                        (dataset.nb_classes * h, nb_per_class * w, 3), dtype="uint8"
                    )

            h_lo = class_id * w
            h_hi = (class_id + 1) * w
            w_lo = sample_id * h
            w_hi = (sample_id + 1) * h

            if shape is not None:
                img = (255 * resize(img, shape)).astype("uint8")

            big_image[h_lo:h_hi, w_lo:w_hi] = img

    plt.imshow(big_image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


def plot(dataset, title="", path=None, nb_per_class=5, shape=None):
    x, _ = sample(dataset, nb_per_class=nb_per_class)
    if dataset.data_type == "image_array" and x.shape[1] == 1:
        x = x.squeeze(1)

    big_image = None
    cmap = None
    for class_id in range(dataset.nb_classes):
        for sample_id in range(nb_per_class):
            if dataset.data_type == "path_array":
                img = dataset.get_image(class_id * nb_per_class + sample_id)
                img = np.asarray(img)
            elif len(x.shape) == 3:  # Grayscale, no channel dimension
                img = x[class_id * nb_per_class + sample_id]
                cmap = "gray"
            else:
                img = x[class_id * nb_per_class + sample_id]

            if big_image is None:
                if shape is None:
                    h = img.shape[0]
                    w = img.shape[1]
                else:
                    h, w = shape

                if cmap == "gray":
                    big_image = np.empty(
                        (dataset.nb_classes * h, nb_per_class * w), dtype=img.dtype
                    )
                else:
                    big_image = np.empty(
                        (dataset.nb_classes * h, nb_per_class * w, 3), dtype="uint8"
                    )

            h_lo = class_id * w
            h_hi = (class_id + 1) * w
            w_lo = sample_id * h
            w_hi = (sample_id + 1) * h

            if shape is not None:
                img = (255 * resize(img, shape)).astype("uint8")

            big_image[h_lo:h_hi, w_lo:w_hi] = img

    plt.imshow(big_image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
