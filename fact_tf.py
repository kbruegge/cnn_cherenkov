import h5py
import tensorflow as tf
import pandas as pd
import numpy as np


def read_n_rows(f, n=10000):
    night = f['events/night'][0:n]
    run = f['events/run'][0:n]
    event = f['events/event'][0:n]
    az = f['events/az'][0:n]
    zd = f['events/zd'][0:n]
    gamma_prediction = f['events/gamma_prediction'][0:n]
    ra_prediction = f['events/ra_prediction'][0:n]
    dec_prediction = f['events/dec_prediction'][0:n]

    df = pd.DataFrame({'night': night, 'run': run, 'event': event, 'zd': zd, 'az': az, 'ra_prediction': ra_prediction, 'dec_prediction': dec_prediction, 'gamma_prediction': gamma_prediction})
    return df, f['events/image'][0:n]


def cnn(images, mode):
    input_layer = images

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    dense = tf.layers.dense(inputs=tf.contrib.layers.flatten(pool), units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Output layer, class prediction
    out = tf.layers.dense(dropout, units=1)
    return tf.nn.softmax(out)


def model_fn(features, labels, mode):
    images = features
    cnn_train = cnn(images, tf.estimator.ModeKeys.TRAIN)
    cnn_predict = cnn(images, tf.estimator.ModeKeys.PREDICT)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=cnn_predict)


    loss = tf.losses.mean_squared_error(labels, cnn_train)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=cnn_predict,
        loss=loss,
        train_op=train_op,)

    return estim_specs


def main():
    f = h5py.File('./crab_images.hdf5')
    df, images = read_n_rows(f, n=5000)

    images = tf.convert_to_tensor(images, np.int16)
    labels = tf.convert_to_tensor(df.gamma_prediction.values, np.float32)
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': images}, y=labels,
        batch_size=128, num_epochs=None, shuffle=True
    )
    print(input_fn)
    model.train(input_fn)
    print(model)


if __name__ == '__main__':
    main()
# Build the Estimator
model = tf.estimator.Estimator(model_fn)
print(model)
