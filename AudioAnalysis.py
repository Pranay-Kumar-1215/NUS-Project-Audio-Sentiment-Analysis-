#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
from pyAudioAnalysis import ShortTermFeatures as sf
import librosa
import librosa.display
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[2]:


file_paths_1 = glob('../Downloads/Audio-Classification-master/Audio_data/*/*.wav')


# In[3]:


neutral_1 = glob('../Downloads/Audio-Classification-master/Audio_data/neutral/*')
angry_1 = glob('../Downloads/Audio-Classification-master/Audio_data/angry/*')

y,sr = librosa.load(angry_1[0])


# In[53]:


neutral_2 = glob('../Downloads/URDU-Dataset-master/Neutral/*')

happy = glob('../Downloads/URDU-Dataset-master/Happy/*')

angry_2 = glob('../Downloads/URDU-Dataset-master/Angry/*')

sad = glob('../Downloads/URDU-Dataset-master/Sad/*')
y,sr = librosa.load(neutral_2[0])


# In[ ]:





# In[907]:


angry = angry_1+angry_2
neutral = neutral_1+neutral_2
negative = angry 
positive = happy
paths = positive +  negative + neutral
len(paths)


# In[ ]:





# In[908]:


#angry
y_angry = librosa.load(angry[0])
y_angry_time = librosa.frames_to_time(y_angry[0])


# In[909]:


spec_angry = librosa.feature.mfcc(y_angry_time)
spec_angry = librosa.amplitude_to_db(spec_angry)

librosa.display.specshow(spec_angry, sr=sr, x_axis='time', y_axis='mel')


# In[910]:


spec_angry = librosa.feature.chroma_stft(y_angry_time)
spec_angry = librosa.amplitude_to_db(spec_angry)
librosa.display.specshow(spec_angry, y_axis='chroma', x_axis='time')

# In[911]:


#neutral
y_neutral,sr = librosa.load(neutral[10])
y_neutral_time = librosa.frames_to_time(y_neutral)


# In[912]:


spec_neutral = librosa.feature.mfcc(y_neutral_time)
spec_neutral = librosa.amplitude_to_db(spec_neutral)
librosa.display.specshow(spec_neutral, sr=sr, x_axis='time', y_axis='mel')

# In[ ]:





# In[913]:


spec_neutral = librosa.feature.chroma_stft(y_neutral)
spec_neutral = librosa.amplitude_to_db(spec_neutral)
librosa.display.specshow(spec_neutral, y_axis='chroma', x_axis='time')



# In[914]:


#sad
y_sad,sr = librosa.load(sad[0])
y_sad_time = librosa.frames_to_time(y_sad)


# In[915]:


spec_sad = librosa.feature.mfcc(y_sad_time)
spec_sad = librosa.amplitude_to_db(spec_sad)
librosa.display.specshow(spec_sad, sr=sr, x_axis='time', y_axis='mel')

# In[916]:


spec_sad = librosa.feature.chroma_stft(y_sad_time)
spec_sad = librosa.amplitude_to_db(spec_sad)
librosa.display.specshow(spec_sad, y_axis='chroma', x_axis='time')


# In[ ]:





# In[ ]:





# In[ ]:





# In[917]:


emotions=[]
for item in positive:
    emotions.append("positive")
for item in neutral:
    emotions.append("neutral")
for item in negative:
    emotions.append("negative")


# In[1012]:


s_neg = 0
for i in range(len(emotions)):
    if(emotions[i]=='negative'):
        s_neg+=1
s_neg


# In[1013]:


s_neut = 0
for i in range(len(emotions)):
    if(emotions[i]=='neutral'):
        s_neut+=1
s_neut


# In[1016]:


s_pos = 0
for i in range(len(emotions)):
    if(emotions[i]=='positive'):
        s_pos+=1
s_pos


# In[921]:


len(emotions)


# In[1022]:


data = {'Positive':s_pos, 'Neutral':s_neut, 'Negative':s_neg}
courses = list(data.keys())
values = list(data.values())


# In[1029]:


# In[ ]:


def load_file(filepath):
    y,sr = librosa.load(filepath)
    return (y,sr)


# In[923]:


def divide_segments(y_time,start):
    # divide into 50 ms segments
    # 50ms -> means 1102 samples    
    y_segmented = y_time[start:start+1111]
    return y_segmented


# In[924]:


def stft(y_seg , sr, window, step):
    [f,f_name] = sf.feature_extraction(y_seg,sr, window, step,  deltas = False)
    return f


# In[925]:


def feature_matricization(y_time):
    y_seg = divide_segments(y_time,0)
    y_stft = stft(y_seg,sr,0.05*sr,0.05*sr)
    y_stft = y_stft.reshape(34)
    
    start = 1111
    y_stfts = y_stft
    while(start <= len(y_time)-1111):
        y_seg = divide_segments(y_time,start)
        #print(y_seg)
        start += 2222
        y_stft = stft(y_seg,sr,0.05*sr,0.05*sr)
        #y_stft_arr = np.asarray(y_stft).T
        y_stft = y_stft.reshape(34)
        y_stfts = np.vstack((y_stfts, y_stft))
        #print(y_stft_arr.shape)

        #np.column_stack((y_stfts, y_stft_arr))
        
    if(len(y_stfts)<70):
        #cal how less
        x = 70 - len(y_stfts)%70
        zeros_arr = np.zeros(shape = (x,34))
        x_arr = np.asanyarray(y_stfts)
        x_arr = x_arr.reshape(-1,34)
        x_final = np.vstack((x_arr,zeros_arr))
        
        return x_final
    if(len(y_stfts)>=70):
        y_stfts = y_stfts[0:70]
        return np.asanyarray(y_stfts)


# In[ ]:





# In[926]:


file_path = paths[0]
label = emotions[0]
labels = np.asarray(label)
y,sr = librosa.load(file_path)
y_time = librosa.frames_to_time(y)
feat = feature_matricization(y_time)
features = feat
for i in range(len(paths)):
    if(i!=0):
        #print(i)
        file_path = paths[i]
        label = emotions[i]
        labels = np.vstack((labels,label))
        y,sr = librosa.load(file_path)
        y_time = librosa.frames_to_time(y)
        feat = feature_matricization(y_time)
        features = np.vstack((features,feat))


# In[927]:


type(features)


# In[928]:


features.shape


# In[929]:


indices = features.shape[0]/70
indices


# In[930]:


features_seg = np.split(features,indices)


# In[931]:


len(features_seg[0][0])


# In[932]:


labels


# In[933]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
le = LabelEncoder()
labels = to_categorical(le.fit_transform(labels))
labels


# In[934]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features_seg,labels)


# In[935]:


len(X_train)


# In[936]:


len(X_test)


# In[937]:


len(y_train)


# In[938]:


len(y_test)


# In[ ]:





# In[939]:


y_train.shape


# In[940]:


X_train_arr = np.asanyarray(X_train)
X_train_arr.shape


# In[ ]:





# In[941]:


X_test_arr = np.asanyarray(X_test)
X_test_arr.shape


# In[942]:


import tensorflow


# In[943]:


from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential


# In[802]:



X_train_arr.shape[1]


# In[1044]:


model = load_model('model_83.h5')


# In[ ]:





# In[ ]:





# In[1048]:


model.fit(X_train_arr,y_train,epochs = 10 , validation_data=(X_test_ar,y_test))


# In[1045]:


model.summary()


# In[1046]:


model.evaluate(X_test_arr,y_test)


# In[1047]:


p = model.predict(X_test_arr)
cm = confusion_matrix(y_test.argmax(axis=1), p.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['positive','neutral','negative'])

disp.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1036]:


y_val = '../Downloads/Audio-Classification-master/Audio_data/test/angry_1.wav'
y_val,sr = librosa.load(y_val)
y_val_time = librosa.frames_to_time(y_val)
feat_matrix_val = feature_matricization(y_val_time)


# In[1038]:





# In[ ]:




