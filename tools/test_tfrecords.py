import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--records_file', type=str, help='Where you store the tfrecords file')
    return parser.parse_args()


def draw_batch(session, images, labels):
    img, lbl = session.run([images, labels])

    img = img.astype(np.uint8)

    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.imshow(img[j, ...])
        plt.title("text {}".format(j))

    plt.show()


if __name__ == '__main__':
    args = load_args()
    data_path = args.records_file

    with tf.Session() as sess:
        feature = {
            'images': tf.FixedLenFeature((), tf.string),
            'imagenames': tf.FixedLenFeature([1], tf.string),
            'labels': tf.VarLenFeature(tf.int64)
        }

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['images'], tf.uint8)

        # Cast label data into int32
        label = tf.cast(features['labels'], tf.int32)

        # Reshape image data into the original shape
        image = tf.reshape(image, [32, 100, 3])

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=10,
            capacity=30,
            num_threads=1,
            min_after_dequeue=10
        )

        # Initialize all global and local variables
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #for batch_index in range(5):
        draw_batch(sess, images, labels)

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
