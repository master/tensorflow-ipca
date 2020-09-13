"""Tests for Incremental PCA."""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np
from sklearn import datasets
from sklearn import decomposition

from .incremental_pca import train_incremental_pca, incremental_pca

BATCHES = 7
BATCH_SIZE = 20
COMPONENTS = 2
FEATURES = 4


class IncrementalPCATest(tf.test.TestCase):
    def test_train_incremental_pca(self):
        iris = datasets.load_iris()
        iris = iris.data[: BATCH_SIZE * BATCHES]
        pca = decomposition.IncrementalPCA(
            n_components=COMPONENTS, batch_size=BATCH_SIZE
        )
        pca.fit(iris)
        proj_ref = pca.transform(iris)
        batch = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, FEATURES])
        step = tf.train.get_or_create_global_step()
        train_op, _ = train_incremental_pca(step, batch, COMPONENTS)
        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            for iris_batch in np.split(iris, BATCHES):
                sess.run(train_op, feed_dict={batch: iris_batch})
            proj = sess.run(incremental_pca(iris, COMPONENTS))
        self.assertAllClose(proj, proj_ref)
