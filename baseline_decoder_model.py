import tensorflow as tf


def covariance(t1, t2, axis=1):
    """Compute covariance between t1 and t2

    Parameters
    ----------
    t1 : tf.Tensor
    t2 : tf.Tensor
    axis : int
        Axis which will be reduced

    Returns
    -------
    tf.Tensor
        The covariance between t1 and t2
    """
    mean_t1 = tf.reduce_mean(t1, axis=axis, keepdims=True)
    mean_t2 = tf.reduce_mean(t2, axis=axis, keepdims=True)

    return tf.matmul(t1 - mean_t1, t2 - mean_t2, transpose_a=True)


def pearson_r(y_true, y_pred, axis=1):
    """Compute the correlation between the true labels and the predicted labels according to an axis

    Parameters
    ----------
    y_true : tf.Tensor
        True labels
    y_pred : tf.Tensor
        Predicted labels
    axis : int
        Selected axis

    Returns
    -------
    tf.Tensor
        Pearson correlation coefficient between y_true and y_pred according to axis
    """
    cov = covariance(y_true, y_pred, axis=axis)
    std1 = tf.sqrt(
        tf.reduce_sum(tf.pow(y_true - tf.reduce_mean(y_true, axis=axis, keepdims=True), 2), axis=axis, keepdims=True))
    std2 = tf.sqrt(
        tf.reduce_sum(tf.pow(y_pred - tf.reduce_mean(y_pred, axis=axis, keepdims=True), 2), axis=axis, keepdims=True))
    return tf.identity(cov / (tf.constant(1e-7) + tf.matmul(std1, std2, transpose_a=True)))


def pearson(t):
    """Wrapper function to use in a Lambda layer to compute the pearson correlation coefficient between two tensors

    Parameters
    ----------
    t : list(tf.Tensor, tf.Tensor)
        list of tensors for which the correlation coefficient should be computed

    Returns
    -------
    tf.Tensor
        Pearson correlation between the tensors
    """
    return pearson_r(t[0], t[1])


def greater(t):
    """Greater than

    Parameters
    ----------
    t : list(tf.Tensor, tf.Tensor)
        List of tensors to compare

    Returns
    -------
    tf.Tensor
        Comparison result: 1 if t[0] > t[1], else 0
    """
    return tf.cast(tf.math.greater(t[0], t[1]), tf.float32)


def sliding_window_wrapper(integration_window_size=16):
    """Construct a sliding window wrapper to compensate for the brain delay.
    We use an integration window size of 16 timesamples, which corresponds to 250ms.

    Parameters
    ----------
    eeg : tf.Tensor
        Input EEG data

    integration_window_size : int
        Integration window size

    Returns
    -------
    function
        Sliding window function

    """

    def sliding_window(eeg):
        return tf.map_fn(
            lambda t: tf.map_fn(
                lambda i: tf.reshape(t[i: i + integration_window_size, :], (64 * integration_window_size,)),
                tf.range(tf.shape(t)[0] - integration_window_size),
                dtype=tf.float32
            ), eeg, dtype=tf.float32)

    return sliding_window


def linear_decoder_baseline(decoder, time_window=640, integration_window_size=16, compile=True):
    """Construct the linear baseline, based on a linear decoder. In our paper, we only used this model for evaluation.

    Parameters
    ----------
    decoder : tf.Model
        Pre-trained decoder model
    time_window : 640
        Segment length
    compile :bool
        If the model should be compiled

    Returns
    -------
    tf.Model
        Linear baseline model
    """
    eeg = tf.keras.layers.Input(shape=[time_window, 64])
    env1 = tf.keras.layers.Input(shape=[time_window, 1])
    env2 = tf.keras.layers.Input(shape=[time_window, 1])

    # Reconstruct the envelope with the decoder
    reconstructed = decoder(eeg)
    # Constructing the integration window at the end of a segment would require samples from outside the segment
    # To prevent this, we discard the last integration_window samples from the segments
    cut_layer = tf.keras.layers.Lambda(lambda t: t[:, :time_window - integration_window_size, :])
    env1_cut = cut_layer(env1)
    env2_cut = cut_layer(env2)

    # Use pearson correlation to compare
    pearson_layer = tf.keras.layers.Lambda(pearson)
    corr1 = pearson_layer([reconstructed, env1_cut])
    corr2 = pearson_layer([reconstructed, env2_cut])

    # Which of the correlation values is bigger?
    out = tf.keras.layers.Flatten()(tf.keras.layers.Lambda(greater)([corr1, corr2]))

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

    if compile:
        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
        print(model.summary())
    return model


def linear_decoder(time_window=640, integration_window_size=16):
    """ A linear decoder, as defined in [1]. We use windows of 10 seconds but truncate the last 250 ms,
    as data from outside the 10 second segment is necessary.


    Parameters
    ----------
    time_window : int
        Length of one segment
    integration_window_size : int
        Size of the integration window

    Returns
    -------
    tf.Model
        A Linear decoder model

    """
    eeg = tf.keras.layers.Input(shape=[time_window, 64])

    # Construct the sliding window
    # Constructing the integration window at the end of a segment would require samples from outside the segment
    # To prevent this, we discard the last integration_window samples from the segments
    eeg_sliding = tf.keras.layers.Lambda(sliding_window_wrapper(integration_window_size))(eeg)

    # Linearly combine samples from the integration window to reconstruct the envelope samples
    reconstructed = tf.keras.layers.Dense(1)(eeg_sliding)

    model = tf.keras.Model(inputs=[eeg], outputs=[reconstructed])

    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=[pearson_r], loss=["mse"])
    print(model.summary())
    return model
