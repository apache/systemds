import sys
import time
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix
from sklearn.preprocessing import RobustScaler
import psutil
import os

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

if __name__ == "__main__":
    input_path = sys.argv[1] + "A.mtx"
    output_path = sys.argv[2] + "B"

    X = mmread(input_path).toarray()

    mem_before = get_memory_usage_mb()

    start_time = time.time()

    # Apply RobustScaler
    scaler = RobustScaler()
    Y = scaler.fit_transform(X)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    mem_after = get_memory_usage_mb()

    
    print("Python (scikit-learn) overall time (ms):")
    print(elapsed_ms)
    print("Memory used during Python execution (MB):")
    print(mem_after - mem_before)

   
    mmwrite(output_path, csc_matrix(Y))
