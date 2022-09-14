import glob, os
import numpy as np
import scipy as sp
from scipy import signal
import tensorflow_io as tfio
import shutil
import matplotlib.pyplot as plt
import random
from pcg_functions import sound_to_spec

#train data
class TrainAugmentation():
    def __init__(self, data_folder, output_folder, output_folder2, list_ids, labels, labels2, classes=[], classes2=[]):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.output_folder2=output_folder2
        self.classes = classes
        self.classes2 = classes2
        self.list_ids = list_ids
        self.labels = labels
        self.labels2 = labels2
        self.output_list_ids = []
        self.output_labels = {}
    
    def __process_npy(self, npy_file):
        root, ext = os.path.splitext(npy_file)
        pid = root.split("/")[-1]
        la = root.split("/")[-1].split("-")[0]
        data = np.load(npy_file)
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        os.makedirs(self.output_folder+'/Absent/', exist_ok=True)
        os.makedirs(self.output_folder+'/Present/', exist_ok=True)
        os.makedirs(self.output_folder+'/Unknown/', exist_ok=True)
        #A=random.randint(1,3)
        
        os.makedirs(self.output_folder2+'/Abnormal/', exist_ok=True)
        os.makedirs(self.output_folder2+'/Normal/', exist_ok=True)
                
        if self.labels[la]==0:
            output_npy_1 = self.output_folder + "/Absent/" + pid + "-1" + ".jpg"
            output_npy_2 = self.output_folder + "/Absent/" + pid + "-2" + ".jpg"
            output_npy_5 = self.output_folder + "/Absent/" + pid + "-3" + ".jpg"
            output_npy_7 = self.output_folder + "/Absent/" + pid + "-4" + ".jpg"
        if self.labels[la]==1:
            output_npy_1 = self.output_folder + "/Present/" + pid + "-1" + ".jpg"
            output_npy_2 = self.output_folder + "/Present/" + pid + "-2" + ".jpg"
            output_npy_5 = self.output_folder + "/Present/" + pid + "-3" + ".jpg"
            output_npy_7 = self.output_folder + "/Present/" + pid + "-4" + ".jpg"
        if self.labels[la]==2:
            output_npy_1 = self.output_folder + "/Unknown/" + pid + "-1" + ".jpg"
            output_npy_2 = self.output_folder + "/Unknown/" + pid + "-2" + ".jpg"
            output_npy_5 = self.output_folder + "/Unknown/" + pid + "-3" + ".jpg"
            output_npy_7 = self.output_folder + "/Unknown/" + pid + "-4" + ".jpg"
        if self.labels2[la]==0:
            output_npy_3 = self.output_folder2 + "/Abnormal/" + pid + "-1" + ".jpg"
            output_npy_4 = self.output_folder2 + "/Abnormal/" + pid + "-2" + ".jpg"
            output_npy_6 = self.output_folder2 + "/Abnormal/" + pid + "-3" + ".jpg"
            output_npy_8 = self.output_folder2 + "/Abnormal/" + pid + "-4" + ".jpg"
        if self.labels2[la]==1:
            output_npy_3 = self.output_folder2 + "/Normal/" + pid + "-1" + ".jpg"
            output_npy_4 = self.output_folder2 + "/Normal/" + pid + "-2" + ".jpg"
            output_npy_6 = self.output_folder2 + "/Normal/" + pid + "-3" + ".jpg"
            output_npy_8 = self.output_folder2 + "/Normal/" + pid + "-4" + ".jpg"
        

        Sx_log = sound_to_spec(np.expand_dims(data, axis = 0), log_spectrogram = True)[2]
        Basic= np.flipud(np.transpose(Sx_log[0]))
        
        BB=plt.subplot(1,1,1)
        BB.imshow(Basic, aspect = 'auto', cmap ='magma')
        BB.grid(False)
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = Basic.shape[1] / fig.dpi 
        Ay = 224 / fig.dpi 
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay)
        fig.canvas.draw()
        data2 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
        black_img=np.zeros((224,224-Basic.shape[1],3))
        concat1=np.concatenate((data2, black_img),axis=1) 
        
        plt.close()
        plt.imshow(concat1.astype('uint8'))
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = 224 / fig.dpi 
        Ay = 224 / fig.dpi      
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay) 
        plt.savefig(output_npy_1)
        plt.savefig(output_npy_3)
        plt.close()
        
        '''
        #Sx_log 일반버전, freq_mask 주파수 마스킹버전, 
        freq_mask = tfio.audio.time_mask(Basic, param=100).numpy()
        BB=plt.subplot(1,1,1)
        BB.imshow(freq_mask, aspect = 'auto', cmap ='magma')
        BB.grid(False)
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = Basic.shape[1] / fig.dpi 
        Ay = 224 / fig.dpi 
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay)
        fig.canvas.draw()
        data3 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data3 = data3.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
        concat2=np.concatenate((data3, black_img),axis=1) 
        plt.close()

        plt.imshow(concat2.astype('uint8'))
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = 224 / fig.dpi 
        Ay = 224 / fig.dpi      
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay) 
        plt.savefig(output_npy_2)
        plt.savefig(output_npy_4)
        plt.close()                 
        '''
        time_mask = tfio.audio.freq_mask(Basic, param=10).numpy()
        BB=plt.subplot(1,1,1)
        BB.imshow(time_mask, aspect = 'auto', cmap ='magma')
        BB.grid(False)
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = Basic.shape[1] / fig.dpi 
        Ay = 224 / fig.dpi 
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay)
        fig.canvas.draw()
        data4 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data4 = data4.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
        concat3=np.concatenate((data4, black_img),axis=1) 
        plt.close()

        plt.imshow(concat3.astype('uint8'))
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = 224 / fig.dpi 
        Ay = 224 / fig.dpi      
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay) 
        plt.savefig(output_npy_2)
        plt.savefig(output_npy_4)
        #plt.savefig(output_npy_5)
        #plt.savefig(output_npy_6)
        plt.close()
            
        '''
        masked = tfio.audio.fade(data, fade_in=10000, fade_out=20000, mode="logarithmic")
        Sx_log_masked = sound_to_spec(np.expand_dims(masked, axis = 0), log_spectrogram = True)[2]
        faded = np.flipud(np.transpose(Sx_log_masked[0]))
        BB=plt.subplot(1,1,1)
        BB.imshow(faded, aspect = 'auto', cmap ='magma')
        BB.grid(False)
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = Basic.shape[1] / fig.dpi 
        Ay = 224 / fig.dpi 
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay)
        fig.canvas.draw()
        data5 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data5 = data5.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
        concat4=np.concatenate((data5, black_img),axis=1) 
        plt.close()

        plt.imshow(concat4.astype('uint8'))
        plt.axis('off') 
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        fig = plt.gcf()
        Ax = 224 / fig.dpi 
        Ay = 224 / fig.dpi      
        fig.set_figwidth(Ax)
        fig.set_figheight(Ay) 
        plt.savefig(output_npy_7)
        plt.savefig(output_npy_8)
        plt.close()
        '''       
        #np.save(output_npy_1, concat1)
        #np.save(output_npy_2, concat2)
        #np.save(output_npy_3, concat3)
        #np.save(output_npy_4, concat4)
                
        self.output_list_ids.append(pid + "-1")
        self.output_list_ids.append(pid + "-2")
        #self.output_list_ids.append(pid + "-3")
        #self.output_list_ids.append(pid + "-4")
       
            
        
    def process(self):
        for f in glob.glob(self.data_folder + "/*.npy"):
            self.__process_npy(f)