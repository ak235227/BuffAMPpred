import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import Bio
from Bio import SeqIO


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Embedding, MaxPool1D, Flatten, ZeroPadding1D, Dropout, \
    Concatenate
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import os

np.random.seed(0)
# print('tensorflow==' + str(tf.__version__))
# print('numpy==' + str(np.__version__))
# print('pandas==' + str(pd.__version__))
# print('biopython==' + str(Bio.__version__))
# print('Python==' + str(sys.version.split(' ')[0]))

# In[2]:

requirement_list = list()
requirement_list.append('Python==' + str(sys.version.split(' ')[0]))
requirement_list.append('tensorflow==' + str(tf.__version__))
requirement_list.append('numpy==' + str(np.__version__))
requirement_list.append('pandas==' + str(pd.__version__))
requirement_list.append('biopython==' + str(Bio.__version__))

# requirement_list.append('sklearn=='+str(sklearn.__version__))
# np.savetxt('../data/requirement.txt', requirement_list, fmt='%s', delimiter=',')
# requirement_list

################ change the file path as you desire ########################
input_data_file = 'input_data_file.txt'
input_format_file = 'input_format.csv'
input_weight_path = 'weights'
output_prediction_file = 'output.csv'
############################################################################

nucleic_acid_list = ['A', 'T', 'G', 'C']
nucleic_acid_dict = dict()
for s in nucleic_acid_list:
    nucleic_acid_dict[s] = nucleic_acid_list.index(s) + 1

amino_acid_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
amino_acid_dict = dict()
for s in amino_acid_list:
    amino_acid_dict[s] = amino_acid_list.index(s) + 1


# print(len(amino_acid_dict), amino_acid_dict)

# In[3]:

def seq_2_array(data_file_path, amino_acid_dict, data_format=None):
    df = pd.DataFrame(columns=['Seq_ID', 'Seq', 'Seq_len', 'Warning'])
    if data_format == 'fasta':
        id_list = list()
        seq_list = list()
        seq_len_list = list()
        warning_list = list()

        for seq_record in SeqIO.parse(data_file_path, "fasta"):
            seq1 = (seq_record.seq).upper()
            #print (list(seq1))
            #print (set(list(seq1)))
            if len(set(list(seq1)).difference(nucleic_acid_dict)) > 0:
                seq = seq1
                #print (seq)
            else:
                seq = str((seq1.translate()).upper()).replace("*", "")
                #print (seq)
        #for seq_record in SeqIO.parse(data_file_path, "fasta"):
            #seq = seq_record.seq.upper()
            id_list.append(seq_record.id)
            seq_list.append(''.join(seq))
            seq_len_list.append(len(seq_record))
            if len(set(list(seq)).difference(amino_acid_dict)) > 0:
                warning_list.append('Sequence contains invalid symbol')
            else:
                warning_list.append('Valid sequence')
        df['Seq_ID'] = id_list
        df['Seq'] = seq_list
        df['Seq_len'] = seq_len_list
        df['Warning'] = warning_list
    df_valid = df.loc[df["Warning"] == 'Valid sequence', :]
    df_invalid = df.loc[df["Warning"] != 'Valid sequence', :]
    # print('valid', df_valid.shape)
    # print('invalid', df_invalid.shape)
    # df.to_csv(r'C:/Users/salin/Downloads/Documents/df.csv')
    # df_valid.to_csv(r'C:/Users/salin/Downloads/Documents/df_valid.csv')
    # df_invalid.to_csv(r'C:/Users/salin/Downloads/Documents/df_invalid.csv')

    window_size = 60
    List = list()
    for i, Seq in enumerate(df_valid['Seq']):
        if len(Seq) <= window_size:
            List.append([df_valid.Seq_ID.values[i], Seq, len(Seq), 'Valid sequence'])
            continue
        Seq_ID = df_valid.Seq_ID.values[i]
        for j in range(len(Seq) - window_size):
            List.append([Seq_ID + '_fragment_' + str(j), Seq[j:j + window_size], window_size, 'Valid sequence'])
    df_new = pd.DataFrame(List, columns=df.columns)
    df_new = df_new.drop_duplicates(subset=['Seq'], keep='first')
    df_new = df_new.reset_index(drop=True)
    # df_new.to_csv(r'C:/Users/salin/Downloads/Documents/df_new.csv')
    max_len = 60 #df_new.Seq_len.max()
    Seq_list = list()
    for seq in df_new["Seq"]:
        Seq_list.append([amino_acid_dict[s] for s in list(seq)])
    Seq_zero_pad = np.array([s + [0] * (max_len - len(s)) for s in Seq_list], dtype=np.int)

    df1 = pd.concat([df_new, df_invalid])
    df1 = df1.sort_values(by="Seq_ID").reset_index(drop=True)
    valid_seq_id = df_valid['Seq_ID'].to_list()
    return Seq_zero_pad, df1


# In[4]:

def create_model(lr=0.001, reg_const=0.00001):
    np.random.seed(0)
    model = Sequential()
    model.add(Embedding(21, 128, input_length=60))
    model.add(Conv1D(filters=64, kernel_size=16, strides=2, padding="valid", activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(LSTM(100, return_sequences=False, return_state=False, use_bias=True, activation="tanh",
                   recurrent_dropout=0.1))
    model.add(Dense(16, kernel_regularizer=l1_l2(reg_const), bias_regularizer=l1_l2(reg_const), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    # model.summary()
    return model


# In[5]:

def prediction_func(data=None, model=None):
    prediction = list()
    for i in range(100):
        weight_file = os.path.join(input_weight_path, 'model_weights' + str(i) + '.h5')
        model.load_weights(weight_file)
        prediction.append(model.predict(data))
    prediction = np.array(prediction, dtype=np.float32)
    prediction = np.mean(prediction, axis=0)
    return prediction