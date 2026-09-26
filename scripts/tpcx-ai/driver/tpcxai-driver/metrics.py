#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


import re
def levenshtein(a, b):
    """
    Calculates the Levenshtein distance between a and b.
    """
    # The following code is from: https://folk.idi.ntnu.no/mlh/hetland_org/coding/python/levenshtein.py
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(y_true, y_pred, pre_process=True):
    """
    Calculates the word error rate
    :param pre_process: If the texts should be pre-processed prior to being used to calculate the WER
    :param y_true: ndarray of true texts
    :param y_pred: ndarray of predicted texts
    :return: the word-error-rate as a float
    """
    def pre_proc(string):
        return re.sub('[^a-z \']', '', string.lower())

    def lev(first_str, second_str):
        first_str = str(first_str)
        second_str = str(second_str)
        if pre_process:
            first_norm = pre_proc(first_str)
            second_norm = pre_proc(second_str)
        else:
            first_norm = first_str.lower()
            second_norm = second_str.lower()
        return levenshtein(first_norm.split(), second_norm.split())
    distances = map(lev, y_true, y_pred)
    lengths = map(lambda txt: len(str(txt).split()), y_true)
    return sum(distances) / sum(lengths)
