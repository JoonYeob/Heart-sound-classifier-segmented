import os
import glob
import numpy as np
from scipy.io import wavfile

def extend_ts(ts, length):
    extended = np.zeros(length)
    siglength = np.min([length, ts.shape[0]])
    extended[:siglength] = ts[:siglength]
    return extended

class Preprocessor():
    def __init__(self, data_folder, output_folder, classes = [], classes2= []):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.list_ids = []
        self.max_length = 50200 #12.55s
        self.classes = classes
        self.classes2 = classes2
        self.labels = {}
        self.labels2 = {}
        self.Ages={}
        self.Sexs={}
        self.Heights={}
        self.Weights={}
        self.Pregs={}
        
    def __process_wav(self, wav_file, murmur, outcome, age, sex, height, weight, preg):   
        #root:path, ext:extension, dir:directory
        root, ext = os.path.splitext(wav_file)
        ID=root.split('/')[-1]
        dirs = "/".join(root.split("/")[:-1])
        frame_starts=None
        frame_ends=None
        trimmed=None
        trimmed2=None
        trimmed3=None
        trimmed4=None
        trimmed5=None
        t2=False
        t3=False
        t4=False
        t5=False

        frequency, recording = wavfile.read(root + '.wav')
        if len(recording)<=self.max_length:
            trimmed = recording
            
        elif len(recording)>self.max_length and len(recording)<=2*self.max_length:
            frame_starts = int((len(recording)-self.max_length)/2)
            frame_ends = int((len(recording)-self.max_length)/2) + self.max_length   
            trimmed = recording[frame_starts:frame_ends]
            
        elif len(recording)>2*self.max_length and len(recording)<=3*self.max_length:
            frame_starts = int((len(recording)-2*self.max_length)/2)
            frame_ends = int((len(recording)-2*self.max_length)/2) + self.max_length   
            trimmed = recording[frame_starts:frame_ends]
            trimmed2= recording[frame_starts+self.max_length : frame_ends+self.max_length ]
            t2=True

        elif len(recording)>3*self.max_length and len(recording)<=4*self.max_length:
            frame_starts = int((len(recording)-3*self.max_length)/2)
            frame_ends = int((len(recording)-3*self.max_length)/2) + self.max_length   
            trimmed = recording[frame_starts:frame_ends]
            trimmed2= recording[frame_starts+self.max_length : frame_ends+self.max_length ]              
            trimmed3= recording[frame_starts+2*self.max_length : frame_ends+2*self.max_length ]
            t2=True
            t3=True
            
        elif len(recording)>4*self.max_length and len(recording)<=5*self.max_length:
            frame_starts = int((len(recording)-4*self.max_length)/2)
            frame_ends = int((len(recording)-4*self.max_length)/2) + self.max_length   
            trimmed = recording[frame_starts:frame_ends]
            trimmed2= recording[frame_starts+self.max_length : frame_ends+self.max_length ]              
            trimmed3= recording[frame_starts+2*self.max_length : frame_ends+2*self.max_length ]
            trimmed4= recording[frame_starts+3*self.max_length : frame_ends+3*self.max_length ]  
            t2=True
            t3=True
            t4=True
            
        elif len(recording)>5*self.max_length and len(recording)<=6*self.max_length:
            frame_starts = int((len(recording)-5*self.max_length)/2)
            frame_ends = int((len(recording)-5*self.max_length)/2) + self.max_length   
            trimmed = recording[frame_starts:frame_ends]
            trimmed2= recording[frame_starts+self.max_length : frame_ends+self.max_length ]              
            trimmed3= recording[frame_starts+2*self.max_length : frame_ends+2*self.max_length ]
            trimmed4= recording[frame_starts+3*self.max_length : frame_ends+3*self.max_length ]              
            trimmed5= recording[frame_starts+4*self.max_length : frame_ends+4*self.max_length ] 
            t2=True
            t3=True
            t4=True
            t5=True
                      
        self.list_ids.append(ID)
        if murmur not in self.classes:
            self.classes.append(murmur)
        if outcome not in self.classes2:
            self.classes2.append(outcome)
        self.labels[ID] = self.classes.index(murmur)
        self.labels2[ID] = self.classes2.index(outcome)
        self.Ages[ID]=age
        self.Sexs[ID]=sex
        self.Heights[ID]=height
        self.Weights[ID]=weight
        self.Pregs[ID]=preg
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        output_npy = self.output_folder + "/" + ID + "-t1.npy"
        output_npy2 = self.output_folder + "/" + ID+ "-t2.npy"
        output_npy3 = self.output_folder + "/" + ID+ "-t3.npy"
        output_npy4 = self.output_folder + "/" + ID+ "-t4.npy"
        output_npy5 = self.output_folder + "/" + ID+ "-t5.npy"
        
        if os.path.isfile(output_npy)==False:
            np.save(output_npy, trimmed)
            if t2 ==True:
                np.save(output_npy2, trimmed2)    
            if t3 ==True:
                np.save(output_npy3, trimmed3)           
            if t4 ==True:
                np.save(output_npy4, trimmed4)               
            if t5 ==True:
                np.save(output_npy5, trimmed5)   
            
    def __process_txt(self, txt_file):
        #root:path, ext:extension, dir:directory
        root, ext = os.path.splitext(txt_file)
        ID=root.split('/')[-1]
        dirs = "/".join(root.split("/")[:-1])
        
        wav_files = []
        pos = ['AV', 'MV', 'TV', 'PV', 'Phc']
        murmurIDs=[]
        nowmurmur=None
        murmur = []
        
        with open(txt_file, "r") as f:
            for line in f:
                splitted = line.strip().split(" ")
                if splitted[0] in pos:
                    for i in splitted[1:]:
                        if i.endswith(".tsv"):
                            wav_files.append(dirs + "/" + i)
                elif splitted[0].startswith("#Murmur:"):
                    nowmurmur = splitted[1]
                elif splitted[1].startswith("locations:"):
                    murmurIDs = splitted[-1].split('+')
                elif splitted[0].startswith("#Outcome:"):
                    Outcome=splitted[1]
                elif splitted[0].startswith("#Age:"):
                    if splitted[1] =='nan':
                        Age = 'Z'
                    else :
                        Age=splitted[1]
                elif splitted[0].startswith("#Sex:"):
                    if splitted[1] =='nan':
                        Sex = 'Z'
                    else:
                        Sex=splitted[1]
                elif splitted[0].startswith("#Height:"):
                    if splitted[1]=='nan':
                        Height = float(115.0)
                    else:
                        Height=float(splitted[1])
                elif splitted[0].startswith("#Weight:"):
                    if splitted[1] =='nan':
                        Weight = float(24.800)
                    else :
                        Weight=float(splitted[1])
                elif splitted[0].startswith("#Pregnancy"):
                    if splitted[-1] =='nan':
                        Preg='Z'
                    else:
                        Preg=splitted[-1]
        for i in range(len(wav_files)):
            murmur.append(nowmurmur)
        for i in range(len(wav_files)):
            self.__process_wav(wav_files[i],murmur[i],Outcome,Age,Sex,Height,Weight,Preg)
                    
    def process(self):
        for f in glob.glob(self.data_folder + "/*.txt"):
            self.__process_txt(f)
