#!/usr/bin/env python

from preprocessor import Preprocessor 
from data_augmentation import TrainAugmentation
from transformers import ViTFeatureExtractor
from datasets import load_metric
from models import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
import io
from pcg_functions import inference_all
from pcg_functions import inferenced_results
from pcg_functions import make_ids_10_fold
from pcg_functions import make_image_folders_10_fold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

from scipy import signal
from tqdm import tqdm 
import PIL.Image
import glob
import matplotlib.pyplot as plt
import shutil
import os
import random
import numpy as np
import tensorflow as tf
from distutils.dir_util import copy_tree

import torch
import datasets
from datasets import Features, Value, ClassLabel, load_dataset, Dataset, DatasetDict, Image, Array3D
from transformers import EarlyStoppingCallback

# Optional TF setting
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')
    # Find the patient data files.
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
    
    pp_folder = "trimmed_npy"
    classes = ['Absent','Present','Unknown']  
    classes2 = ['Abnormal', 'Normal']
    
    train_folder = 'train_npy'
    val_folder = 'val_npy'
    
    train_aug = 'train_aug' 
    val_aug = 'val_aug'
    
    train_out = 'train_out'
    val_out = 'val_out'
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    os.makedirs(train_aug, exist_ok=True)
    os.makedirs(val_aug, exist_ok=True)
    
    print('Preprocessing start')
    pp = Preprocessor(data_folder, pp_folder, classes)
    pp.process()
    print('Preprocessing end')

    list_ids = pp.list_ids
    labels = pp.labels
    lables2 = pp.labels2
    n_classes = len(pp.classes)
    n_classes2 = len(pp.classes2)
    
    new_dict={}
    for key in pp.Pregs.keys():
        new_dict[key]=[pp.Ages[key], pp.Sexs[key], pp.Heights[key], pp.Weights[key], pp.Pregs[key]]
    df = pd.DataFrame.from_dict(new_dict, orient='index', columns=['Ages','Sexs','Heights','Weights','Pregs'])
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    df['Ages']=encoder.fit_transform(df['Ages'])
    np.save('Ages_classes.npy', encoder.classes_)
    df['Sexs']=encoder.fit_transform(df['Sexs'])
    np.save('Sexs_classes.npy', encoder.classes_)
    df['Pregs']=encoder.fit_transform(df['Pregs'])
    np.save('Pregs_classes.npy', encoder.classes_)
    df[['Heights', 'Weights']] =scaler.fit_transform(df[['Heights', 'Weights']])
    df=df.reset_index()   
   
    unique_list, unique_ids = make_ids_10_fold(list_ids)
    
    train_folder_list = ['./train_'+str(i)+'/' for i in range(1,11)]
    val_folder_list = ['./val_'+str(i)+'/' for i in range(1,11)]
    otrain_folder_list = ['./otrain_'+str(i)+'/' for i in range(1,11)]
    oval_folder_list = ['./oval_'+str(i)+'/' for i in range(1,11)]
    
    print('Making image start')
    train_augmented_im = TrainAugmentation(data_folder=pp_folder, output_folder=train_aug, output_folder2=train_out, list_ids=unique_ids, labels=pp.labels, labels2 = pp.labels2, classes=len(pp.classes))
    
    train_augmented_im.process()
    print('Making image done')

    print('10-fold')  
    make_image_folders_10_fold(train_folder_list, val_folder_list, otrain_folder_list, oval_folder_list, unique_list, unique_ids, list_ids)
        
    if verbose >= 1:
        print('Training model...')
    
    # Train the murmur
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def transform(example_batch):
        filenamelist=list()
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        inputs['meta'] = [torch.FloatTensor(x) for x in example_batch['meta']]       
        inputs['labels'] = example_batch['label']
        return inputs

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch]),
            'meta' : torch.stack([x['meta'] for x in batch])
        }
    metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    def compute_f1_metrics(p):
        return f1_metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='weighted')
    
    final_loss=0
    for i in range(10): 
        
        print(i+1, "th training start")
                
        train_ds = load_dataset("imagefolder", data_dir=(train_folder_list[i]), split="train", ignore_verifications=True, save_infos=True)
        train_ds = train_ds.cast_column("image", Image())
        
        val_ds = load_dataset("imagefolder", data_dir=(val_folder_list[i]), split="train", ignore_verifications=True, save_infos=True)
        val_ds = val_ds.cast_column("image", Image())

        ds = DatasetDict({"train":train_ds, "validation":val_ds})
        ds=ds.map(lambda x: {"meta": df[df['index']==(x['image'].filename).split('/')[-1].split('-')[0]].values.tolist()[0][1:]},
    batched=False)
        
        prepared_ds = ds.with_transform(transform)
        
        labels = ds['train'].features['label'].names

        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
        )

        training_args = TrainingArguments(
          output_dir="./vit-base-beans",
          per_device_train_batch_size=64,
          evaluation_strategy="steps",
          num_train_epochs=100,
          fp16=True,
          save_steps=100,
          eval_steps=100,
          logging_steps=5,
          learning_rate=0.0001,
          save_total_limit=2,
          remove_unused_columns=False,
          push_to_hub=False,
          report_to='tensorboard',
          load_best_model_at_end=True,
        )

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")

                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 3.0])).to(device)

                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=model,
            #optimizers=(optimizer,scheduler),
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["validation"],
            tokenizer=feature_extractor,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        
        metrics = trainer.evaluate(prepared_ds['validation'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        del train_ds
        del val_ds
        del ds
        del prepared_ds
        
        if i==0:
            # For the first fold, unconditionally save the model and the validation loss.
            torch.save(model,'kfoldmodel.pt')
            final_loss=metrics['eval_loss']
        else:
            #For the rest of the fold, it is overwritten only when the val loss of the current model is smaller than the stored loss (because the memory is initialized when the model declaration is renewed every time), in this case final_loss is the current loss.
            if metrics['eval_loss']< final_loss:
                torch.save(model,'kfoldmodel.pt')
                final_loss = metrics['eval_loss']
    
    model = torch.load('kfoldmodel.pt')
    
    #outcome
    final_loss_2=0
    for i in range(10): 
        
        print(i+1, "th training start")
    
        train_ds2 = load_dataset("imagefolder", data_dir=(otrain_folder_list[i]), split="train", save_infos=True, ignore_verifications=True)
        train_ds2 = train_ds2.cast_column("image", Image())

        val_ds2 = load_dataset("imagefolder", data_dir=(oval_folder_list[i]), split="train", save_infos=True, ignore_verifications=True)
        val_ds2 = val_ds2.cast_column("image", Image())

        ds2 = DatasetDict({"train":train_ds2, "validation":val_ds2})
        ds2=ds2.map(lambda x: {"meta": df[df['index']==(x['image'].filename).split('/')[-1].split('-')[0]].values.tolist()[0][1:]},
        batched=False)
        prepared_ds2 = ds2.with_transform(transform)

        labels2 = ds2['train'].features['label'].names

        model2 = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(labels2),
        )

        training_args2 = TrainingArguments(
          output_dir="./vit-base-beans",
          per_device_train_batch_size=32,
          evaluation_strategy="steps",
          num_train_epochs=20,
          fp16=True,
          save_steps=50,
          eval_steps=50,
          logging_steps=5,
          learning_rate=0.0001,
          save_total_limit=2,
          remove_unused_columns=False,
          push_to_hub=False,
          report_to='tensorboard',
          load_best_model_at_end=True,
        )
        
        class CustomTrainer2(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.0])).to(device)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        '''
        trainer2 = Trainer(
            model=model2,
            args=training_args2,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds2["train"],
            eval_dataset=prepared_ds2["validation"],
            tokenizer=feature_extractor,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]

        )
        '''
        trainer2 = CustomTrainer2(
            model=model2,
            args=training_args2,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds2["train"],
            eval_dataset=prepared_ds2["validation"],
            tokenizer=feature_extractor,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]

        )
        
        train_results2 = trainer2.train()
        trainer2.save_model()
        trainer2.log_metrics("train", train_results2.metrics)
        trainer2.save_metrics("train", train_results2.metrics)
        trainer2.save_state()
        
        metrics2 = trainer2.evaluate(prepared_ds2['validation'])
        trainer2.log_metrics("eval", metrics2)
        trainer2.save_metrics("eval", metrics2)
        
        del train_ds2
        del val_ds2
        del ds2
        del prepared_ds2
        
        if i==0:
            torch.save(model2,'kfoldoutputmodel.pt')
            final_loss_2=metrics2['eval_loss']
        else:
            if metrics2['eval_loss']< final_loss_2:
                torch.save(model2,'kfoldoutputmodel.pt')
                final_loss_2 = metrics2['eval_loss']

    model2 = torch.load('kfoldoutputmodel.pt')
       
    # Save model.
    PATH = model_folder + "/model.pt"

    torch.save({'murmur_classifier': model,
                'outcome_classifier': model2
                }, PATH)
    
    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.pt')
    print(filename)
    return torch.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    
    pred_arr, pred_arr2 = inference_all(model, './', recordings, data)
    labels, probabilities = inferenced_results(pred_arr, pred_arr2)
        
    classes = ['Present', 'Unknown', 'Absent', 'Abnormal','Normal']
    return classes, labels, probabilities
