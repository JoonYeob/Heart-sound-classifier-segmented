FROM huggingface/transformers-pytorch-gpu:4.20.1
#FROM tensorflow/tensorflow:2.9.1-gpu

#docker run --rm --gpus all -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER author@example.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

## Include the following line if you have a requirements.txt file.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
