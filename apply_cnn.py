from cnn_fact import cnn_model_fn
from cnn_fact import scale_images
import image_io
import numpy as np
import tensorflow as tf
import click


@click.command()
@click.argument('images', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_file', type=click.Path(exists=False, dir_okay=False))
def main(images, out_file):
    path = './data/crab_images.hdf5'

    print('Total number of images: {}'.format(image_io.number_of_images(path)))
    df, images = image_io.read_n_rows(path, start=0, end=-1)
    eval_data = scale_images(images)
    eval_labels = df.gamma_prediction.values.astype(np.float32)

    print('image shape: {}'.format(images.shape))

    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: cnn_model_fn(features, labels, mode, learning_rate=0.001),
        model_dir='./data/convnet_model',
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    predictions = list(mnist_classifier.predict(input_fn=eval_input_fn))
    predictions = [p['probabilities'][0] for p in predictions]
    df['predictions_convnet'] = predictions
    df.to_hdf(out_file, key='events')


if __name__ == "__main__":
    main()
