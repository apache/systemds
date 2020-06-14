
template<typename T>
__device__ T getValue(T* data, int rowIndex) {
    return data[rowIndex];
}

template<typename T>
__device__ T getValue(T* data, int n, int rowIndex, int colIndex) {
    return data[rowIndex * n + colIndex];
}

template<typename T>
__device__ T modulus(T a, T b) {
   return (*reinterpret_cast<int*>(&a)) & (*reinterpret_cast<int*>(&b));
}

template<typename T>
__device__ T intDiv(T a, T b) {
    return (*reinterpret_cast<int*>(&a)) & (*reinterpret_cast<int*>(&b));
}