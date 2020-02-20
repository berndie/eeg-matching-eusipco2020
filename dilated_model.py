import tensorflow as tf


def dilated_model(time_window=640, layers=3, kernel_size=3, spatial_filters=8, dilated_filters=16, activation="relu",
                  compile=True, inputs=tuple(), output_name="output"):
    """Convolutional dilated model

    Parameters
    ----------
    time_window : int
        Segment length
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilated convolutions
    spatial_filters : int
        Number of parallel filters to use in the spatial layer
    dilated_filters : int
        Number of parallel filters to use in the dilated layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilated layers or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs
    output_name : str
        Name to give to the output
    Returns
    -------
    tf.Model
        The dilated model
    """

    # If different inputs are required
    if len(inputs) == 3:
        eeg, env1, env2 = inputs[0], inputs[1], inputs[2]
    else:
        eeg = tf.keras.layers.Input(shape=[time_window, 64])
        env1 = tf.keras.layers.Input(shape=[time_window, 1])
        env2 = tf.keras.layers.Input(shape=[time_window, 1])

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    env_proj_1 = env1
    env_proj_2 = env2
    # Spatial convolution
    eeg_proj_1 = tf.keras.layers.Conv1D(spatial_filters, kernel_size=1)(eeg)

    # Construct dilated layers
    for l in range(layers):
        # dilated on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(dilated_filters, kernel_size=kernel_size, dilated_rate=kernel_size ** l,
                                            strides=1, activation=activations[l])(eeg_proj_1)

        # dilated on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(dilated_filters, kernel_size=kernel_size, dilated_rate=kernel_size ** l,
                                                strides=1, activation=activations[l])
        env_proj_1 = env_proj_layer(env_proj_1)
        env_proj_2 = env_proj_layer(env_proj_2)

    # Comparison
    cos1 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_1])
    cos2 = tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, env_proj_2])

    # Classification
    out1 = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos1, cos2])))

    # 1 output per batch
    out = tf.keras.layers.Reshape([1], name=output_name)(out1)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    if compile:
        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
        print(model.summary())
    return model
