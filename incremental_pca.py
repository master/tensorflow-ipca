"""Incremental PCA."""
import tensorflow.compat.v1 as tf


def train_incremental_pca(step, inputs, n_components):
    """Implement the incremental PCA model from:

    D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, May 2008.
    See http://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

    Args:
      step: Training step variable.
      inputs: A float32 `Tensor` of input data.
      n_components: Number of components to keep.

    Returns:
      A tuple of `noise_variance` and `train_op` `Tensor`-s, where
        `noise_variance` is the estimated noise covariance following the
        Probabilistic PCA model from Tipping and Bishop 1999.
        See "Pattern Recognition and Machine Learning" by C. Bishop, 12.2.1
        p. 574 or http://www.miketipping.com/papers/met-mppca.pdf.
    """
    with tf.variable_scope('IncrementalPCA', [inputs]):
        n_samples, n_features = inputs.shape

        n_samples_seen = tf.get_variable(
            'n_samples_seen',
            [1],
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
        )
        running_mean = tf.get_variable(
            'running_mean', [1, n_features], initializer=tf.zeros_initializer()
        )
        components = tf.get_variable(
            'components',
            [n_components, n_features],
            initializer=tf.zeros_initializer(),
        )
        singular_vals = tf.get_variable('singular_vals', [n_components])

        n_total_samples = tf.cast(n_samples_seen + n_samples, tf.float32)

        col_mean = running_mean * tf.to_float(n_samples_seen)
        col_mean += tf.reduce_sum(inputs, -2, keepdims=True)
        col_mean /= n_total_samples

        col_batch_mean = tf.reduce_mean(inputs, -2, keepdims=True)

        mean_correction = tf.sqrt(
            tf.to_float(
                (n_samples_seen * n_samples) / (n_samples_seen + n_samples)
            )
        ) * (running_mean - col_batch_mean)

        x = tf.concat(
            [
                tf.reshape(singular_vals, [-1, 1]) * components,
                inputs - col_batch_mean,
                mean_correction,
            ],
            axis=0,
        )

        s, _, v = tf.svd(x, full_matrices=False, compute_uv=True)

        v = -tf.transpose(v)
        abs_v = tf.abs(v)
        m = tf.equal(abs_v, tf.reduce_max(abs_v, axis=-2, keepdims=True))
        m = tf.cast(m, v.dtype)
        signs = tf.sign(tf.reduce_sum(v * m, axis=-2, keepdims=True))
        v *= signs

        explained_variance = tf.square(s) / (n_total_samples - 1)
        noise_variance = tf.reduce_mean(explained_variance[n_components:])

        with tf.control_dependencies(
            [
                components.assign(v[:n_components]),
                singular_vals.assign(s[:n_components]),
            ]
        ):

            train_op = tf.group(
                n_samples_seen.assign_add([n_samples]),
                running_mean.assign(col_mean),
                step.assign_add(1),
                name='train_op',
            )
            return train_op, noise_variance


def incremental_pca(inputs, n_components):
    """Incremental PCA transform.

    Args:
      inputs: A float32 `Tensor` of input data.
      n_components: Number of components to keep.

    Returns:
      A `Tensor` of transformed data.
    """
    with tf.variable_scope('IncrementalPCA', [inputs], reuse=tf.AUTO_REUSE):
        n_features = inputs.shape[1]
        components = tf.get_variable('components', [n_components, n_features])
        running_mean = tf.get_variable('running_mean', [1, n_features])
        return tf.matmul((inputs - running_mean), tf.transpose(components))
