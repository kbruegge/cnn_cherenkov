import numpy as np
import tensorflow as tf
import image_io
from sklearn.preprocessing import StandardScaler


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, learning_rate=0.01):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 45, 46, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 45, 46, 32]
    # Output Tensor Shape: [batch_size, 22, 23, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # # Convolutional Layer #2
    # # Computes 64 features using a 5x5 filter.
    # # Padding is added to preserve width and height.
    # # Input Tensor Shape: [batch_size, 20, 18, 32]
    # # Output Tensor Shape: [batch_size, 20, 18, 64]
    # conv2 = tf.layers.conv2d(
    #     inputs=pool1,
    #     filters=64,
    #     kernel_size=[5, 5],
    #     padding="same",
    #     activation=tf.nn.relu)
    #
    # # Pooling Layer #2
    # # Second max pooling layer with a 2x2 filter and stride of 2
    # # Input Tensor Shape: [batch_size, 22, 23, 64]
    # # Output Tensor Shape: [batch_size, 11, 11, 64]
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 10, 9, 64]
    # Output Tensor Shape: [batch_size, 10 * 9 * 64]
    pool1_flat = tf.reshape(pool1, [-1, 22 * 23 * 32])

    # Dense Layer
    # Densely connected layer with 100 neurons
    # Input Tensor Shape: [batch_size, 10 * 9 * 64]
    # Output Tensor Shape: [batch_size, 100]
    dense = tf.layers.dense(
        inputs=pool1_flat, units=100, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 100]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(inputs=dropout, units=1)

    predictions = {
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.mean_squared_error(tf.squeeze(labels), tf.squeeze(predictions['probabilities']))
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'mse': tf.metrics.mean_squared_error(labels, predictions['probabilities']),
        'mae': tf.metrics.mean_absolute_error(labels, predictions['probabilities']),
        'rmse': tf.metrics.root_mean_squared_error(labels, predictions['probabilities']),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def scale_images(images):
    s = images.shape
    X = images.astype(np.float32).reshape(len(images), -1)
    return StandardScaler().fit_transform(X).reshape(s)


def main(argv):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # print("mnist shape: {}".format(train_data.shape))

    path = './data/crab_images.hdf5'

    print('Total number of images: {}'.format(image_io.number_of_images(path)))
    df, images = image_io.read_n_rows(path, start=0, end=130000)

    train_data = scale_images(images)
    train_labels = df.gamma_prediction.values.astype(np.float32)

    df, images = image_io.read_n_rows(path, start=130000, end=138000)

    eval_data = scale_images(images)
    eval_labels = df.gamma_prediction.values.astype(np.float32)

    print('image shape: {}'.format(train_data.shape))

    # Create the Estimator
    # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    #     eval_data,
    #     eval_labels,
    #     every_n_steps=50
    # )

    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: cnn_model_fn(features, labels, mode, learning_rate=0.001),
        model_dir='./data/convnet_model'
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=2,
        shuffle=True,
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        # monitors=[validation_monitor],
    )
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
