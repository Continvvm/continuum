import matplotlib.pyplot as plt
import numpy as np


def sample(x, y, nb_per_class=5):
    sampled_x, sampled_y = [], []

    for class_id in np.unique(y):
        indexes = np.where(y == class_id)[0][:nb_per_class]
        sampled_x.append(x[indexes])
        sampled_y.append(y[indexes])
    sampled_x = np.concatenate(sampled_x)
    sampled_y = np.concatenate(sampled_y)

    return sampled_x, sampled_y


def plot(dataset, figsize=None, path=None, nb_per_class=5):
    x, y = sample(dataset.x, dataset.y, nb_per_class=nb_per_class)
    if not dataset.open_image and x.shape[1] == 1:
        x = x.squeeze(1)

    c = 1
    for class_id in range(dataset.nb_classes):
        for sample_id in range(nb_per_class):
            ax = plt.subplot(dataset.nb_classes, nb_per_class, c)
            if dataset.open_image:
                img = dataset.get_image(class_id * nb_per_class + sample_id)
                ax.imshow(np.asarray(img))
            elif len(x.shape) == 3:  # Grayscale, no channel dimension
                ax.imshow(x[class_id * nb_per_class + sample_id], cmap="gray")
            else:
                ax.imshow(x[class_id * nb_per_class + sample_id])

            ax.set_xticks([])
            ax.set_yticks([])
            c += 1

    plt.subplots_adjust(bottom=0.01, top=0.5, wspace=0.00, hspace=0.0001)
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
