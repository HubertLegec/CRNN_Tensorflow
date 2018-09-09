# CRNN - TensorFlow
Use TensorFlow to implement a Deep Neural Network for scene text recognition mainly based on the paper "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition".
You can refer to their paper for details [http://arxiv.org/abs/1507.05717](http://arxiv.org/abs/1507.05717). Thanks for the author [Baoguang Shi](https://github.com/bgshih).  
This model consists of a CNN stage, RNN stage and CTC loss for scene text recognition task.

## Installation

### Standard way
All the required packages you may install by executing command.
```bash
pip install -r requirements.txt
```
To do that you need python 3.6 and pip to be installed on your machine.

### Docker
To build docker container with prepared environment execute following command:  
```bash
docker build -t hlegec/crnn:1.0 .
```

## Project

### Training
To train the model run following command:
```bash
python train.py -c config.yaml -d input_dir
```

### Testing
To evaluate trained model run following command:
```bash
python test.py -c config.yaml -d input_dir -w weights_path
```

### Prediction
To do prediction for a single image execute:
```bash
python demo.py -i data/test_3/word_930.png -w model/crnn_dsc_2018-08-20.ckpt -c src/config.yaml
```

