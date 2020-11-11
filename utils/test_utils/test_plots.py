import numpy as np
import tensorflow as tf

from ..random import random_indices
from ..plot_utils import grid_plot

__all__ = ['test_grid_plot_mnist']


def test_grid_plot_mnist():

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = x_test[..., None].astype('float32')
    m_test = len(x_test)

    random_state = 42

    nrows = 16
    ncols = 12

    label = np.random.randint(0, 11)

    sample_size = nrows * ncols
    indices = random_indices(n=sample_size, start=0, end=m_test, step=1, replace=False, random_state=random_state)

    images = x_test[indices]

    true_i = np.where(y_test == label)[0]
    ground_truth = x_test[true_i][:nrows]

    grid_plot(images=images, nrows=nrows, ncols=ncols, ground_truth=ground_truth)
