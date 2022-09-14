# Heart-sound-classifier-segmented
Heart sound classifier using the vision transformer model which is code for the George B. Moody PhysioNet Challenge 2022.
This code is one of the version using a segmented sound.

# Run code in python

```
  pip install requirements.txt
```
### Put your training data in training_data folder and run next code.
```
  python train_model.py training_data model
```
### Put your test data in test_data folder. 
### The model folder will contain a model.pt file.

```
  python run_model.py model test_data test_outputs
  python evaluate_model.py test_data test_outputs
```
### You can see results.


# Run code in docker

### This script is from https://github.com/physionetchallenges/python-classifier-2022

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data test_data model test_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2022). Dividing them training data in `training_data` and test_data in `test_data`. 

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/JoonYeob/Heart-sound-classifier-segmented.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        model  Heart-sound-classifier-segmented  test_data  test_outputs  training_data

        user@computer:~/example$ cd Heart-sound-classifier-segmented/

        user@computer:~/example/Heart-sound-classifier-segmented$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/Heart-sound-classifier-segmented$ docker run -it -v ~/example/model:/physionet/model -v ~/example/test_data:/physionet/test_data -v ~/example/test_outputs:/physionet/test_outputs -v ~/example/training_data:/physionet/training_data image bash

        root@[...]:/physionet# python train_model.py training_data model

        root@[...]:/physionet# python run_model.py model test_data test_outputs

        root@[...]:/physionet# python evaluate_model.py test_data test_outputs
        [...]

        root@[...]:/physionet# exit
        Exit
