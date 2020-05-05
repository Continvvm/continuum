import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize


def sample(x, y, nb_per_class=5):
    sampled_x, sampled_y = [], []

    for class_id in np.unique(y):
        indexes = np.where(y == class_id)[0][:nb_per_class]
        sampled_x.append(x[indexes])
        sampled_y.append(y[indexes])
    sampled_x = np.concatenate(sampled_x)
    sampled_y = np.concatenate(sampled_y)

    return sampled_x, sampled_y


def plot(dataset, title="", path=None, nb_per_class=5, shape=None):
    x, _ = sample(dataset.x, dataset.y, nb_per_class=nb_per_class)
    if not dataset.open_image and x.shape[1] == 1:
        x = x.squeeze(1)

    big_image = None
    cmap = None
    for class_id in range(dataset.nb_classes):
        for sample_id in range(nb_per_class):
            if dataset.open_image:
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
