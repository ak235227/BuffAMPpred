from _datetime import date, datetime
import json
import xlsxwriter
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow .keras.models import load_model, Sequential
from tensorflow .keras.layers import Input, Dense, LSTM, Conv1D, Embedding, MaxPool1D, Flatten, ZeroPadding1D, Dropout, \
    Concatenate
from tensorflow .keras.regularizers import l1_l2, l1, l2
from tensorflow .keras.optimizers import RMSprop, Adam, SGD
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv
import time, os, io, shutil, pathlib, os.path, glob
import uuid
import cnn_lstm_model_predict
from flask import Flask, request, jsonify, render_template, flash, redirect,request
from flask import url_for, send_from_directory, send_file, after_this_request, current_app, session
from werkzeug.utils import secure_filename
import pickle
from werkzeug.datastructures import FileStorage

nof = 5

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/var/www/html/aamir_bufamp_ws/data_file' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# filename = str(uuid.uuid4())
rad1 = str(uuid.uuid4()) #str(int(random.random() * 1000))
output_path = os.path.join(app.config['UPLOAD_FOLDER'], rad1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/Tutorial_manual')
def Tutorial_manual():
    manual = "Manual.pdf"
    return send_file(manual, as_attachment=True)

@app.route('/downld', methods = ['GET','POST'])
def downld():
    rnd=request.form['ranid']
    path='data_file/'+str(rnd)+'/output.csv'
    return send_file(path, as_attachment=True)

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/BuffAMPdb')
def BuffAMPdb():
    return render_template('contact.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/uploader1', methods=['POST'])
def uploader1():
    global path, flag, path1
    if request.method == 'POST':
        # f = request.files['seqfile']
        print('%%%%%%%%%%%% File Data %%%%%%%%%%%%')
        print(request.files)
        print(request.form)
        print('%%%%%%%%%%%% File Data End %%%%%%%%%%%%')
        seqtxt = request.form.get('sequence')
        print('%%%%%%%%%%%% File Data %%%%%%%%%%%%')
        print(seqtxt)
        print('%%%%%%%%%%%% File Data End %%%%%%%%%%%%')
        rad1 = str(uuid.uuid4()) #str(int(random.random() * 1000))
        path = os.path.join(app.config['UPLOAD_FOLDER'], rad1)
        os.mkdir(path)
        flag = 0
        if 'seqfile' in request.files:
            f = request.files['seqfile']
            flag = 1
            f.filename = "input_dat_file.txt"
            filename = secure_filename(f.filename)
            f.save(os.path.join(path, filename))
            path1 = os.path.join(path, "input_dat_file.txt")
            print(path1)
        else:
            flag = 2
            path1 = os.path.join(path, "input_dat_file.txt")
            print(path1)
            f1 = open(path1, "w")
            f1.write(seqtxt)
            f1.close()
    output_prediction_file = os.path.join(path, 'output.csv')
    amino_acid_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y']
    amino_acid_dict = dict()
    input_format = 'fasta'
    for s in amino_acid_list:
        amino_acid_dict[s] = amino_acid_list.index(s) + 1
    #print(len(amino_acid_dict), amino_acid_dict)
    if flag >= 1:
        Seq_zero_pad, df = cnn_lstm_model_predict.seq_2_array(data_file_path=path1, amino_acid_dict=amino_acid_dict,
                                                              data_format=input_format)
        model = cnn_lstm_model_predict.create_model()
        prediction_prob = cnn_lstm_model_predict.prediction_func(data=Seq_zero_pad, model=model).T[0]
        prediction_class = prediction_prob > 0.5
        prediction_prob[np.where(prediction_class == False)] = 1 - prediction_prob[np.where(prediction_class == False)]
        prediction_class = np.array(['AMP' if x == True else 'non_AMP' for x in prediction_class], dtype=np.str)
        prediction_prob = np.asarray(prediction_prob)
        prediction_class = np.asarray(prediction_class)
        df["Prediction_class"] = None
        #df["Prediction_prob"] = None
        df.loc[df['Warning'] == 'Valid sequence', 'Prediction_class'] = prediction_class
        #df.loc[df['Warning'] == 'Valid sequence', 'Prediction_prob'] = prediction_prob
        #df_ = df.sort_values(by="Prediction_prob", ascending=False).reset_index(drop=True)
        #print(df_)
        df.to_csv(output_prediction_file, index=False)
        return render_template("analysis1.html", name1=rad1,
                               res=output_prediction_file,
                               tables=[df.to_html(classes='data', header='true')])
    else:
        return 'Wrong inputs, Please do needful'

def remove_thing(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def empty_directory(path):
    for i in glob.glob(os.path.join(path, '*')):
        remove_thing(i)

empty_directory('/var/www/html/aamir_bufamp_ws/data_file/') #deletes contents, folders & files only is files are not readonly

if __name__ == '__main__':
    # app.run(debug = True)
    app.run(host='0.0.0.0', debug=True, port=5030)
    # app.run(host='0.0.0.0', debug=True, port=4444)
