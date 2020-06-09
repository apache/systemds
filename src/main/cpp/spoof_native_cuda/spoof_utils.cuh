template<typename T>
__device__ T getValue(T* data, int rowIndex) {
    return data[rowIndex];
}

template<typename T>
__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {
    return data[rowIndex * n + colIndex];
}