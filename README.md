# CRNN_Tensorflow
Use tensorflow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".
You can refer to their paper for details http://arxiv.org/abs/1507.05717. Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation

### Standard way
All the required packages you may install by executing command.
```bash
pip3 install -r requirements.txt
```
To do that you need python 3.6 and pip to be installed on your machine.

### Docker
To build docker container with prepared environment execute following command:  
```bash
docker build -t hlegec/crnn:1.0 .
```

## Test model
In this repo I uploaded a model trained on [Synth 90k](http://www.robots.ox.ac.uk/~vgg/data/text/). 
During data preparation process the dataset is converted into a tensorflow records.
You can test the trained model on the converted dataset by
```bash
python test.py -d new_data/ -w model/shadownet/shadownet_2018-07-19-11-53-37.ckpt-39999 -c src/config.yaml
```

`Expected output is`  
![Test output](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/test_output.png)

If you want to test a single image you can do it by
```bash
python demo.py -i data/test_images/test_06.jpg -w model/shadownet/shadownet_2018-07-19-11-53-37.ckpt-39999 -c src/config.yaml
```

## Train your own model
#### Data Preparation
Firstly you need to store all your image data in some folder then you need to supply at least one of the following text files 
to specify the relative path to the image data dir and it's corresponding text label: `annotation_val.txt`, `annotation_test.txt`, `annotation_train.txt`.
For example

```
path/1/2/373_coley_14845.jpg coley
path/17/5/176_Nevadans_51437.jpg nevadans
```

Secondly you are supposed to convert your dataset into tensorflow records which can be done by
```
python write_text_features -d path/to/your/dataset -d path/to/tfrecords_dir
```
Each of the training images will be scaled into (32, 100, 3). The dataset will be divided into train, test, validation set according to provided files.

#### Train model
The whole training epochs are 40000 in my experiment. I trained the model with a batch size 32, initialized learning rate is 0.1 and decrease by multiply 0.1 every 10000 epochs.
For more training parameters information you can check the global_configuration/config.py for details. To train your own model by

```
python train.py --dataset_dir path/to/your/tfrecords
```
You can also continue the training process from the snapshot by
```
python train.py --dataset_dir path/to/your/tfrecords --weights_path path/to/your/last/checkpoint
```
After several times of iteration you can check the log file in logs folder you are supposed to see the following contenent
![Training log](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_log.png)
The seq distance is computed by calculating the distance between two saparse tensor so the lower the accuracy value is the better the model performs.
The train accuracy is computed by calculating the character-wise precision between the prediction and the ground truth so the higher the better the model performs.

During my experiment the `loss` drops as follows  
![Training loss](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/train_loss.png)
The `distance` between the ground truth and the prediction drops as follows  
![Sequence distance](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/seq_distance.png)

## Experiment
The accuracy during training process rises as follows  
![Training accuracy](https://github.com/TJCVRS/CRNN_Tensorflow/blob/master/data/images/training_accuracy.md)
