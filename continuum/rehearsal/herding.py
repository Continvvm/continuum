from typing import Any

import torch
import numpy as np


def herd_random(x: np.ndarray, y: np.ndarray, t: np.ndarray, z: Any, nb_per_class: int) -> np.ndarray:
    """Herd randomly examples for rehearsal.

    :param x: Input data (images, paths, etc.)
    :param y: Labels of the data.
    :param t: Task ids of the data.
    :param z: Extra info, here unused.
    :param nb_per_class: Number of samples to herd per class.
    :return: The sampled data x, y, t.
    """
    indexes = []

    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        indexes.append(
            np.random.choice(
                class_indexes,
                size=min(nb_per_class, len(class_indexes)),
                replace=False
            )
        )

    indexes =  np.concatenate(indexes)
    return x[indexes], y[indexes], t[indexes]


def herd_closest_to_cluster(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    features: np.ndarray,
    nb_per_class: np.ndarray
) -> np.ndarray:
    """Herd the samples whose features is the closest to their class mean.

    :param x: Input data (images, paths, etc.)
    :param y: Labels of the data.
    :param t: Task ids of the data.
    :param features: Features of shape (nb_samples, nb_dim).
    :param nb_per_class: Number of samples to herd per class.
    :return: The sampled data x, y, t.
    """
    if len(features.shape) != 2:
        raise ValueError(f"Expected features to have 2 dimensions, not {len(features.shape)}d.")
    indexes = []

    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        class_features = features[class_indexes]
        class_mean = np.mean(class_features, axis=1, keepdims=True)

        dist_to_mean = np.linalg.norm(class_mean - class_features, axis=1)
        tmp_indexes = dist_to_mean.argsort()[:nb_per_class]

        indexes.append(class_indexes[tmp_indexes])

    indexes =  np.concatenate(indexes)
    return x[indexes], y[indexes], t[indexes]


def herd_closest_to_barycenter(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    features: np.ndarray,
    nb_per_class: int
) -> np.ndarray:
    """Herd the samples whose features is the closest to their moving barycenter.

    Reference:
        * iCaRL: Incremental Classifier and Representation Learning
          Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
          CVPR 2017

    :param x: Input data (images, paths, etc.)
    :param y: Labels of the data.
    :param t: Task ids of the data.
    :param features: Features of shape (nb_samples, nb_dim).
    :param nb_per_class: Number of samples to herd per class.
    :return: The sampled data x, y, t.
    """
    if len(features.shape) != 2:
        raise ValueError(f"Expected features to have 2 dimensions, not {len(features.shape)}d.")

    indexes = []

    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        class_features = features[class_indexes]

        D = class_features.T
        D = D / (np.linalg.norm(D, axis=0) + 1e-8)
        mu = np.mean(D, axis=1)
        herding_matrix = np.zeros((class_features.shape[0],))

        w_t = mu
        iter_herding, iter_herding_eff = 0, 0

        while not (
            np.sum(herding_matrix != 0) == min(nb_per_class, class_features.shape[0])
        ) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                iter_herding += 1

            w_t = w_t + mu - D[:, ind_max]

        herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

        tmp_indexes = herding_matrix.argsort()[:nb_per_class]
        indexes.append(class_indexes[tmp_indexes])

    indexes =  np.concatenate(indexes)
    return x[indexes], y[indexes], t[indexes]
