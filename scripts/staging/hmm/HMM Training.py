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

# String generation using Hidden Markov Model
#Author: Afan Secic


import numpy as np
import pandas as pd
import random
from itertools import combinations

def add2dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

def list2probabilitydict(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict

def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key

def generate_generic(sentence, no_of_words_to_generate = 1, previous_words = 3):
    sentence = sentence.split()
    if len(sentence) < previous_words:
        previous_words = len(sentence)
    if len(sentence) == 0:
        sentence.append(sample_word(initial_word))
        no_of_words_to_generate = no_of_words_to_generate - 1
    if len(sentence) == 1:
        word0 = sentence[0]
        if word0 in second_word.keys():
            word1 = sample_word(second_word[word0])
        else:
            word1 = np.random.choice(list(second_word[word0].keys()), 1, p = list(second_word[word0].values()))[0]
        sentence.append(word1)
        no_of_words_to_generate = no_of_words_to_generate - 1
    
    while no_of_words_to_generate > 0:
        existing_keys = []
        previous_words_temp = previous_words
        found_keys = False
        while previous_words_temp != 0:
            words = list(combinations(sentence, previous_words_temp))
            previous_words_temp = previous_words_temp - 1
            existing_keys = list(set(words).intersection(transitions))
            if(len(existing_keys) != 0):
                found_keys = True
                break
        if found_keys:
            existing_keys = np.array(existing_keys)
            chosen_key = tuple(existing_keys[np.random.choice(len(existing_keys),1)][0])
            word = np.random.choice(list(transitions[chosen_key].keys()), 1, p = list(transitions[chosen_key].values()))[0]
            sentence.append(word)
            no_of_words_to_generate = no_of_words_to_generate - 1
        else:
            chosen_key = np.random.choice(list(transitions.keys()), 1)[0]
            word = np.random.choice(list(transitions[chosen_key].keys()), 1, p = list(transitions[chosen_key].values()))[0]
            sentence.append(word)
            no_of_words_to_generate = no_of_words_to_generate - 1
        
    print(' '.join(sentence))

def train_markov_model_generic(data, no_of_words):
    if no_of_words > 3:
        no_of_words = 3
    for line in data:
        line_length = len(line)
        first_token = line[0]
        initial_word[first_token] = initial_word.get(first_token, 0) + 1
        for i in range(1,line_length-1):
            for j in range(len(line[:i+1]) if len(line[:i+1]) < no_of_words + 1 else no_of_words + 1):
                word_combinations = combinations(line[:i+1], j)
                for combination in list(word_combinations):
                    if len(combination) > 0:    
                        if i == 1:
                            add2dict(second_word, combination if len(combination) > 1 else combination[0], line[i+1])
                        else:
                            add2dict(transitions, combination if len(combination) > 1 else combination[0], line[i+1])

    initial_word_total = sum(initial_word.values())
    for key, value in initial_word.items():
        initial_word[key] = value / initial_word_total
        
    for prev_word, next_word_list in second_word.items():
        second_word[prev_word] = list2probabilitydict(next_word_list)
        
    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = list2probabilitydict(next_word_list)

data = pd.read_csv('text_matrix.csv', dtype=type('string') ,header=None).values
data = np.array([row[~pd.isnull(row)] for row in data])
initial_word = {}
second_word = {}
transitions = {}

#Second parameter is to determine how many previous words algorithm takes when learning
train_markov_model_generic(data, 5)
sentence = 'drought smith say'
generate_generic(sentence)