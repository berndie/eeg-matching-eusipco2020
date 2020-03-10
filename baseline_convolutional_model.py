"""Default simple convolutional model."""
import tensorflow as tf


def simple_convolutional_model(
        time_window,
        filters=1,
        kernel_size=16,
        channels=64
):
    """Construct the simple convolutional model.

    Parameters
    ----------
    time_window: int
        Time window of input data in samples
    filters: int
        Number of filters for the convolutional layer
    kernel_size: int
        Kernel size for the convolutional layer
    channels: int
        Number of channels in the EEG


    Returns
    -------
    tf.keras.model.Model
        Simple convolutional model
    """
    # If different inputs are required
    eeg = tf.keras.layers.Input([time_window, channels])
    env1 = tf.keras.layers.Input([time_window, 1])
    env2 = tf.keras.layers.Input([time_window, 1])

    eeg_proj = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size)(eeg)

    cut_layer = tf.keras.layers.Lambda(lambda t: t[:, :-(kernel_size-1), :])
    dot_layer = tf.keras.layers.Dot(1, normalize=True)
    cos1 = dot_layer([eeg_proj, cut_layer(env1)])
    cos2 = dot_layer([eeg_proj, cut_layer(env2)])

    all_cos = tf.keras.layers.Concatenate()([cos1, cos2])
    flat = tf.keras.layers.Flatten()(all_cos)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc"],
        loss=["binary_crossentropy"]
    )
    print(model.summary())
    return model
