import numpy as np
import tensorflow as tf
import image_io
from sklearn.preprocessing import StandardScaler
import click

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, learning_rate=0.01, n_classes=10):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = features["x"]

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[10, 10],
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
    # # Input Tensor Shape: [batch_size, 22, 23, 32]
    # # Output Tensor Shape: [batch_size, 22, 23, 64]
    conv2 = tf.layers.conv2d(
         inputs=pool1,
         filters=64,
         kernel_size=[5, 5],
         padding="same",
         activation=tf.nn.relu)

    # # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.flatten(pool2)

    # Dense Layer
    # Input Tensor Shape: [batch_size, 10 * 9 * 64]
    # Output Tensor Shape: [batch_size, 100]
    dense_1 = tf.layers.dense(
        inputs=pool2_flat, units=500, activation=tf.nn.relu)

    dense_2 = tf.layers.dense(
        inputs=dense_1, units=100, activation=tf.nn.relu)
    # Add dropout operation; 0.9 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense_2, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 100]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(inputs=dropout, units=n_classes)

    predictions = {
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities':  tf.nn.softmax(logits, name="softmax_tensor"),
        'classes': tf.argmax(logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    f_labels = tf.to_float(labels, name='labels')
    weights = tf.add(tf.divide(f_labels, tf.to_float(tf.reduce_sum(labels)) / tf.to_float(tf.size(labels))), tf.constant(1.0), name='weights')
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=labels, depth=n_classes, name='one_hot')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    tf.Print(labels, [predictions['probabilities'], labels])

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),
        #'roc_auc': tf.metrics.auc(labels=labels, predictions=predictions['probabilities']),
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def scale_images(images):
    X = images.astype(np.float32).reshape(len(images), -1)
    return StandardScaler().fit_transform(X).reshape((len(images), 45, 46, -1))


@click.command()
@click.option('-l', '--learning_rate', default=0.001)
@click.option('--mnist/--no-mnist', default=False)
@click.option('-c', '--n_classes', default=10)
def main(mnist, learning_rate, n_classes):
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
    session = tf.Session(config=config)
    with session:
        if not mnist:
            path = './data/crab_images.hdf5'

            print('Total number of images: {}'.format(image_io.number_of_images(path)))
            df, images = image_io.read_n_rows(path, start=0, end=130000)

            bins = np.linspace(0, 1, 2, endpoint=False)

            train_data = scale_images(images)
            train_labels = df.gamma_prediction.values.astype(np.float32)
            train_labels = np.digitize(train_labels, bins=bins) - 1

            df, images = image_io.read_n_rows(path, start=130000, end=138000)

            eval_data = scale_images(images)
            eval_labels = df.gamma_prediction.values.astype(np.float32)
            eval_labels = np.digitize(eval_labels, bins=bins) - 1
            print('image shape: {}'.format(train_data.shape))
            model_path='./data/fact_model'
        else:
            #Load training and eval data
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            train_data = mnist.train.images
            train_data = train_data.reshape((len(train_data), 28, 28, -1))
            train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
            eval_data = mnist.test.images
            eval_data = eval_data.reshape((len(eval_data), 28, 28, -1))
            eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
            print("mnist shape: {}".format(train_data.shape))
            model_path='./data/mnist_model'


        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
                model_fn=lambda features, labels, mode: cnn_model_fn(features, labels, mode, learning_rate=learning_rate, n_classes=n_classes), model_dir=model_path)

        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {'one_hot':'one_hot', 'labels':'labels'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=256,
            num_epochs=2,
            shuffle=True,
        )
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=100,
            hooks=[logging_hook],
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


        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            num_epochs=1,
            shuffle=False,
        )

        predictions = list(mnist_classifier.predict(input_fn=eval_input_fn))
        predictions = [p['probabilities'] for p in predictions]
        print('prediction')
        print(np.histogram(predictions, bins=np.linspace(0, 1, 20)))


if __name__ == "__main__":
    main()
