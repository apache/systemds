#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

from bs4 import BeautifulSoup,SoupStrainer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np
import csv
import os

def create_dataset(input_file_name, output_file_name):
    f = open(input_file_name, 'r')
    data= f.read()
    lemmatizer = WordNetLemmatizer() 
    soup = BeautifulSoup(data,'html.parser')
    sentences = []
    text_matrix = []
    for item in soup.findAll('body'):
        sentences = sent_tokenize(item.text)
        for sentence in sentences:
            text_matrix.append([token for token in word_tokenize(sentence) if token.lower() not in stopwords.words('english') and token not in string.punctuation])
    for i in range(len(text_matrix)):
        for j in range(len(text_matrix[i])):
            text_matrix[i][j] = lemmatizer.lemmatize(text_matrix[i][j].lower(), pos='v')
    length = max(map(len, text_matrix))
    text_matrix=np.array([row + [None] * (length - len(row)) for row in text_matrix])
    try:
        with open(output_file_name) as csvFile:
            os.remove(output_file_name)
            csvFile = open(output_file_name, 'w')
            for row in text_matrix:
                writer = csv.writer(csvFile)
                writer.writerow(row)
        csvFile.close() 
    except IOError:
        with open(output_file_name, 'w') as csvFile:
            for row in text_matrix:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
            csvFile.close() 

create_dataset('reut2-000.sgm', 'text_matrix.csv')