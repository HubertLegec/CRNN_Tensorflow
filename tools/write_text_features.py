import os
import os.path as ops
import argparse

from data_provider import TfRecordBuilder


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset')
    parser.add_argument('--save_dir', type=str, help='Where you store tfrecords')

    return parser.parse_args()


def write_features(dataset_dir, save_dir):
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    # write val tfrecords
    print('Start writing validation tf records')

    val_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_val.txt'), ops.join(save_dir, 'validation_feature.tfrecords'))
    val_builder.process()

    # write test tfrecord
    print('Start writing testing tf records')

    test_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_test.txt', ops.join(save_dir, 'test_feature.tfrecords')))
    test_builder.process()

    # write train tfrecords
    print('Start writing training tf records')
    train_builder = TfRecordBuilder(ops.join(dataset_dir, 'annotation_train.txt'), ops.join(save_dir, 'train_feature.tfrecords'))
    train_builder.process()

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir)
