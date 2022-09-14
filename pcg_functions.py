from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import io
from scipy import signal
from tqdm import tqdm 
import PIL.Image
import glob
import matplotlib.pyplot as plt
import shutil
import os
import random
import numpy as np
from distutils.dir_util import copy_tree
import torch
from transformers import EarlyStoppingCallback
from helper_code import *

def sound_to_spec(data, nperseg=446, noverlap=223, log_spectrogram = True):
    #Convert hurt sound to spectrogram/log scale spectrogram with overlap ratio = 0.5
    fs = 4000
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nperseg)
    Sxx = np.transpose(Sxx,[0,2,1])
    if log_spectrogram:
        Sxx = abs(Sxx) 
        mask = Sxx > 0 
        Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def inference_all(model, now_location, recordings, data):
    '''
    input 
    model: trained model
    now_location: path
    recordings: .wav files (signal data)
    data: .txt files (metadata)
    output
    pred_arr: murmur class results for each signal
    pred_arr2: outcome class results for each signal
    '''
    classes = ['Absent','Present','Unknown']
    n_classes = 3
    classes2 = ['Abnormal','Normal']
    n_classes2 = 2
    max_length = 50200 #max_length=12.55s  

    Ages = get_age(data)
    Sexs = get_sex(data)
    Heights = get_height(data)
    Weights = get_weight(data)
    Pregs = str(get_pregnancy_status(data))    
    #In case of missing values, it is filled in with the mode or a new class 
    if Ages=='nan':
        Ages='Child'
    if Sexs=='nan':
        Sexs='Female'
    if Heights=='nan':
         Heights=float(115.0)
    if Weights=='nan':
        Weights=float(24.800)
    if Pregs=='nan':
        Pregs=='False'
    
    #Label encoding
    encoder = LabelEncoder()
    #encoder.classes_ = np.load('Ages_classes.npy', allow_pickle=True)
    encoder.classes_ = np.array(['Adolescent', 'Child', 'Infant', 'Neonate', 'Z'], dtype=object)
    Ages_list=[]
    Ages_list.append(Ages)
    Ages=encoder.transform(Ages_list).tolist()[0]
    
    Sexs_list=[]
    Sexs_list.append(Sexs)
    encoder2 = LabelEncoder()
    encoder2.classes_ = np.array(['Female', 'Male', 'Z'], dtype=object)    
    Sexs=encoder2.transform(Sexs_list).tolist()[0]

    #Minmax scaling
    Heights  = (Heights - 35) / (180-35)
    Weights = (Weights - 2.3) / (110.8 - 2.3)
    
    Pregs_list=[]
    Pregs_list.append(Pregs)
    encoder3 = LabelEncoder()
    encoder3.classes_ = np.array(['False', 'True', 'Z'], dtype=object)
    Pregs=encoder3.transform(Pregs_list).tolist()[0]

    #meta_list: preprocessed metadata
    meta_list = []
    meta_list.append(Ages)
    meta_list.append(Sexs)
    meta_list.append(Heights)
    meta_list.append(Weights)
    meta_list.append(Pregs)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
   
    pred_arr=[]
    pred_arr2=[]
    
    for record in recordings:
        #trimmed_arr: array of segmented sound signals
        trimmed_arr=list()
        
        if len(record)<=max_length:
            trimmed1 = record
            trimmed_arr.append(trimmed1)
            
        elif len(record)>max_length and len(record)<=2*max_length:
            frame_starts = int((len(record)-max_length)/2)
            frame_ends = int((len(record)-max_length)/2) + max_length   
            trimmed1 = record[frame_starts:frame_ends]
            trimmed_arr.append(trimmed1)

        elif len(record)>2*max_length and len(record)<=3*max_length:
            frame_starts = int((len(record)-2*max_length)/2)
            frame_ends = int((len(record)-2*max_length)/2) + max_length   
            trimmed1 = record[frame_starts:frame_ends]
            trimmed2 = record[frame_starts+max_length:frame_ends+max_length]
            trimmed_arr.append(trimmed1)
            trimmed_arr.append(trimmed2)
        
        elif len(record)>3*max_length and len(record)<=4*max_length:
            frame_starts = int((len(record)-3*max_length)/2)
            frame_ends = int((len(record)-3*max_length)/2) + max_length   
            trimmed1 = record[frame_starts:frame_ends]
            trimmed2 = record[frame_starts+max_length:frame_ends+max_length]            
            trimmed3 = record[frame_starts+2*max_length:frame_ends+2*max_length] 
            trimmed_arr.append(trimmed1)
            trimmed_arr.append(trimmed2)
            trimmed_arr.append(trimmed3)
            
        elif len(record)>4*max_length and len(record)<=5*max_length:
            frame_starts = int((len(record)-4*max_length)/2)
            frame_ends = int((len(record)-4*max_length)/2) + max_length   
            trimmed1 = record[frame_starts:frame_ends]
            trimmed2 = record[frame_starts+max_length:frame_ends+max_length]            
            trimmed3 = record[frame_starts+2*max_length:frame_ends+2*max_length] 
            trimmed4 = record[frame_starts+3*max_length:frame_ends+3*max_length]
            trimmed_arr.append(trimmed1)
            trimmed_arr.append(trimmed2)
            trimmed_arr.append(trimmed3)
            trimmed_arr.append(trimmed4)
            
        elif len(recording)>5*max_length and len(recording)<=6*max_length:
            frame_starts = int((len(recording)-5*max_length)/2)
            frame_ends = int((len(recording)-5*max_length)/2) + max_length   
            trimmed1 = recording[frame_starts:frame_ends]
            trimmed2= recording[frame_starts+max_length : frame_ends+max_length ]              
            trimmed3= recording[frame_starts+2*max_length : frame_ends+2*max_length ]
            trimmed4= recording[frame_starts+3*max_length : frame_ends+3*max_length ]              
            trimmed5= recording[frame_starts+4*max_length : frame_ends+4*max_length ] 
            trimmed_arr.append(trimmed1)
            trimmed_arr.append(trimmed2)
            trimmed_arr.append(trimmed3)
            trimmed_arr.append(trimmed4)
            trimmed_arr.append(trimmed5)
            
        for trimmed in trimmed_arr:
            Spec = sound_to_spec(np.expand_dims(trimmed, axis = 0), log_spectrogram = True)[2]
            Basic= np.flipud(np.transpose(Spec[0]))        
            Spec_image=plt.subplot(1,1,1)
            Spec_image.imshow(Basic, aspect = 'auto', cmap ='magma')
            Spec_image.grid(False)
            plt.axis('off') 
            plt.tight_layout()
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
            fig = plt.gcf()
            Ax = Basic.shape[1] / fig.dpi 
            Ay = 224 / fig.dpi 
            fig.set_figwidth(Ax)
            fig.set_figheight(Ay)
            fig.canvas.draw()

            Spec_image_2 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            Spec_image_2 = Spec_image_2.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
            black_img=np.zeros((224,224-Basic.shape[1],3))
            Spec_concated=np.concatenate((Spec_image_2, black_img),axis=1) 
            plt.close()

            plt.imshow(Spec_concated.astype('uint8'))
            plt.axis('off') 
            plt.tight_layout()
            plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
            fig = plt.gcf()
            Ax = 224 / fig.dpi 
            Ay = 224 / fig.dpi      
            fig.set_figwidth(Ax)
            fig.set_figheight(Ay) 
            plt.savefig(now_location+'/test.jpg')
            plt.close()

            img=PIL.Image.open(now_location + '/test.jpg', 'r')

            #encoding = feature_extractor(images=np.array(img)[:,:,:3], return_tensors="pt")
            encoding = feature_extractor(images=img, return_tensors="pt")
            encoding.keys()
            pixel_values = encoding['pixel_values'].to(device)
            meta_now=torch.FloatTensor(meta_list)

            final_value=torch.cat((model['murmur_classifier'].vit(pixel_values)[0][0][0],meta_now.to(device)))
            final_value2=torch.cat((model['outcome_classifier'].vit(pixel_values)[0][0][0],meta_now.to(device)))

            outputs = model['murmur_classifier'].classifier(final_value)
            outputs2 = model['outcome_classifier'].classifier(final_value2)
            prediction = outputs.argmax(-1)
            pred_arr.append(prediction)

            prediction2 = outputs2.argmax(-1)  
            pred_arr2.append(prediction2)    

    return pred_arr, pred_arr2

def inferenced_results(pred_arr, pred_arr2):
    if 1 in pred_arr:
        labels=np.array([1,0,0]) # Present, the priority is high, so if at least one is present: present        
    if 2 in pred_arr and 1 not in pred_arr:
        labels=np.array([0,1,0]) # Unknown
    if 1 not in pred_arr and 2 not in pred_arr:
        labels=np.array([0,0,1]) # Absent
    n0=0
    n1=0
    for i in pred_arr2:
        if i==0:
            n0+=1
        if i==1:
            n1+=1
    if n0>0:
        labels=np.concatenate((labels, np.array([1,0])))
    if n0==0:
        labels=np.concatenate((labels, np.array([0,1])))
    probabilities=list()
    probabilities.append(pred_arr.count(1)/len(pred_arr))
    probabilities.append(pred_arr.count(2)/len(pred_arr))
    probabilities.append(pred_arr.count(0)/len(pred_arr))
    probabilities.append(n0/(n0+n1))
    probabilities.append(n1/(n0+n1))    
    probabilities=np.array(probabilities)
    return labels, probabilities

def make_ids_10_fold(list_ids):
    #Create dataset for cross validation
    unique_ids = list(set([id.split("_")[0] for id in list_ids]))
    random.seed(21)
    random.shuffle(unique_ids)
    
    #10-fold
    unique_1 = unique_ids[:int(len(unique_ids)*0.1)]
    unique_2 = unique_ids[int(len(unique_ids)*0.1):int(len(unique_ids)*0.2)]
    unique_3 = unique_ids[int(len(unique_ids)*0.2):int(len(unique_ids)*0.3)]
    unique_4 = unique_ids[int(len(unique_ids)*0.3):int(len(unique_ids)*0.4)]
    unique_5 = unique_ids[int(len(unique_ids)*0.4):int(len(unique_ids)*0.5)]
    unique_6 = unique_ids[int(len(unique_ids)*0.5):int(len(unique_ids)*0.6)]
    unique_7 = unique_ids[int(len(unique_ids)*0.6):int(len(unique_ids)*0.7)]
    unique_8 = unique_ids[int(len(unique_ids)*0.7):int(len(unique_ids)*0.8)]
    unique_9 = unique_ids[int(len(unique_ids)*0.8):int(len(unique_ids)*0.9)]
    unique_10 = unique_ids[int(len(unique_ids)*0.9):]
    
    for i in range(1,11):
        os.makedirs('./train_'+str(i), exist_ok=True)
        os.makedirs('./val_'+str(i), exist_ok=True)
        os.makedirs('./otrain_'+str(i), exist_ok=True)
        os.makedirs('./oval_'+str(i), exist_ok=True)
        
    unique_list = [unique_1,unique_2,unique_3,unique_4,unique_5,unique_6,unique_7,unique_8,unique_9,unique_10]
    
    return unique_list, unique_ids

def make_image_folders_10_fold(train_folder_list, val_folder_list, otrain_folder_list, oval_folder_list, unique_list, unique_ids, list_ids):
    for number in range(10):
        
        now_train_folder = train_folder_list[number]
        now_val_folder = val_folder_list[number]
        
        now_otrain_folder = otrain_folder_list[number]
        now_oval_folder = oval_folder_list[number]
        
        os.makedirs(now_val_folder+'/Absent', exist_ok=True)
        os.makedirs(now_val_folder+'/Present', exist_ok=True)
        os.makedirs(now_val_folder+'/Unknown', exist_ok=True)
        os.makedirs(now_oval_folder+'/Abnormal', exist_ok=True)
        os.makedirs(now_oval_folder+'/Normal', exist_ok=True)
        
        # Copies all data to the train folder of the corresponding fold, and in the case of files corresponding to the val set, the augmentation is deleted, and only if it is not, it is moved to the val folder.
        
        val_unique=unique_list[number]
        train_unique = list(set(unique_ids) - set(val_unique))
        train_ids = [ id for id in list_ids if id.split("_")[0] in train_unique]
        val_ids = [ id for id in list_ids if id.split("_")[0] in val_unique ]
        copy_tree('train_aug', now_train_folder)
        copy_tree('train_out', now_otrain_folder)
        for j in glob.glob(now_train_folder+'/Absent/*'):
            if j.split('/')[-1].split('-')[0] in val_ids:
                if j.split('-')[-1]=='1.jpg':
                    shutil.move(j, now_val_folder+'/Absent/')
                if j.split('-')[-1]=='2.jpg':
                    os.remove(j)   
                if j.split('-')[-1]=='3.jpg':
                    os.remove(j)
                if j.split('-')[-1]=='4.jpg':
                    os.remove(j)                 
        for j in glob.glob(now_train_folder+'/Present/*'):
            if j.split('/')[-1].split('-')[0] in val_ids:
                if j.split('-')[-1]=='1.jpg':
                    shutil.move(j, now_val_folder+'/Present/')
                if j.split('-')[-1]=='2.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='3.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='4.jpg':
                    os.remove(j) 
        for j in glob.glob(now_train_folder+'/Unknown/*'):
            if j.split('/')[-1].split('-')[0] in val_ids:
                if j.split('-')[-1]=='1.jpg':
                    shutil.move(j, now_val_folder+'/Unknown/')
                if j.split('-')[-1]=='2.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='3.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='4.jpg':
                    os.remove(j) 
        for j in glob.glob(now_otrain_folder+'/Abnormal/*'):
            if j.split('-')[-1]=='2.jpg':
                os.remove(j) 
            if j.split('/')[-1].split('-')[0] in val_ids:
                if j.split('-')[-1]=='1.jpg':
                    shutil.move(j, now_oval_folder+'/Abnormal/')
                '''
                if j.split('-')[-1]=='2.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='3.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='4.jpg':
                    os.remove(j)
                '''
        for j in glob.glob(now_otrain_folder+'/Normal/*'):
            if j.split('-')[-1]=='2.jpg':
                os.remove(j) 
            if j.split('/')[-1].split('-')[0] in val_ids:
                if j.split('-')[-1]=='1.jpg':
                    shutil.move(j, now_oval_folder+'/Normal/')
                '''
                if j.split('-')[-1]=='2.jpg':
                    os.remove(j) 
                if j.split('-')[-1]=='3.jpg':
                    os.remove(j)  
                if j.split('-')[-1]=='4.jpg':
                    os.remove(j) 
                '''
