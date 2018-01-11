from cnn_fact import cnn_model_fn
from cnn_fact import scale_images
import image_io
import numpy as np
import tensorflow as tf
import click
import matplotlib.pyplot as plt

@click.command()
@click.argument('images', type=click.Path(exists=True, dir_okay=False))
@click.argument('out_file', type=click.Path(exists=False, dir_okay=False))
@click.option('-c', '--n_classes', default=2)
def main(images, out_file, n_classes):
    path = './data/crab_images.hdf5'

    print('Total number of images: {}'.format(image_io.number_of_images(path)))
    df, images = image_io.read_n_rows(path, start=0, end=40000)
    eval_data = scale_images(images)
    eval_labels = df.gamma_prediction.values.astype(np.float32)

    print('image shape: {}'.format(images.shape))

    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: cnn_model_fn(features, labels, mode, n_classes=n_classes),
        model_dir='./data/fact_model',
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

    plt.hist(predictions, bins=np.linspace(0, 1, 15), label='prediction', alpha=0.5)
    plt.hist(eval_labels, bins=np.linspace(0, 1, 15), label='truth', alpha=0.5)
    plt.legend()
    plt.savefig('dist_compare.pdf')
    plt.figure()
    plt.plot(predictions, eval_labels, '.')
    plt.savefig('correlation.pdf')

    df['predictions_convnet'] = predictions
    df.to_hdf(out_file, key='events')


if __name__ == "__main__":
    main()
