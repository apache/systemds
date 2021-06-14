/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once
#ifndef SYSTEMDS_MATRIX_H
#define SYSTEMDS_MATRIX_H

using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
struct Matrix {
	int32_t nnz;
	uint32_t rows;
	uint32_t cols;
	
	uint32_t* row_ptr;
	uint32_t* col_idx;
	T* data;
	
	typedef T value_type;
	
	explicit Matrix(size_t* jvals) : nnz(jvals[0]), rows(jvals[1]), cols(jvals[2]),
			row_ptr(reinterpret_cast<uint32_t*>(jvals[3])),
			col_idx(reinterpret_cast<uint32_t*>((jvals[4]))), data(reinterpret_cast<T*>(jvals[5])) {}
};

#ifdef __CUDACC__

template<typename T>
uint32_t bin_search(T* values, uint32_t lower, uint32_t upper, T val) {
	upper -= 1;
	while(lower <= (upper-1)) {
		uint32_t idx = (lower + upper) >> 1;
		uint32_t vi = values[idx];
		if (vi < val)
			lower = idx + 1;
		else {
			if (vi <= val)
				return idx;
			upper = idx - 1;
		}
	}
	return upper + 1;
}

template<typename T>
class MatrixAccessor {
	
	Matrix<T>* _mat;

public:
	MatrixAccessor() = default;
	
	__device__ MatrixAccessor(Matrix<T>* mat) : _mat(mat) {}
	
	__device__ void init(Matrix<T>* mat) { _mat = mat; }
	
	__device__ uint32_t& nnz() { return _mat->nnz; }
	__device__ uint32_t cols() { return _mat->cols; }
	__device__ uint32_t rows() { return _mat->rows; }
	
	__device__ uint32_t len() { return _mat->row_ptr == nullptr ? len_dense() : len_sparse(); }
	
	__device__ uint32_t pos(uint32_t rix) {
		return _mat->row_ptr == nullptr ? pos_dense(rix) : pos_sparse(rix);
	}
	
	__device__ T& val(uint32_t r, uint32_t c) {
		return _mat->row_ptr == nullptr ? val_dense_rc(r,c) : val_sparse_rc(r, c) ;
	}
	
	__device__ T& val(uint32_t i) {
		return _mat->row_ptr == nullptr ? val_dense_i(i) : val_sparse_i(i);
	}
	__device__ T& operator[](uint32_t i) { return val(i); }
	
	__device__ T* vals(uint32_t rix) {
		return _mat->row_ptr == nullptr ? vals_dense(rix) : vals_sparse(rix) ;
	}
	
	__device__ uint32_t row_len(uint32_t rix) {
		return _mat->row_ptr == nullptr ? row_len_dense(rix) : row_len_sparse(rix);
	}
	
	__device__ uint32_t* col_idxs(uint32_t rix) { return cols_sparse(rix); }

	__device__ void set(uint32_t r, uint32_t c, T v) { set_sparse(r,c,v); }
	
	__device__ uint32_t* indexes() {  return _mat->row_ptr;	}
	
	__device__ bool hasData() { return _mat->data != nullptr; }
private:
	__device__ uint32_t len_dense() {
		return _mat->rows * _mat->cols;
	}
	
	__device__ uint32_t pos_dense(uint32_t rix) {
		return _mat->cols * rix;
	}
	
	__device__ T& val_dense_rc(uint32_t r, uint32_t c) {
		return _mat->data[_mat->cols * r + c];
	}
	
	__device__ T& val_dense_i(uint32_t i) {
		return _mat->data[i];
	}
	
	__device__ T* vals_dense(uint32_t rix) {
		return &(_mat->data[rix]);
	}
	
	__device__ uint32_t row_len_dense(uint32_t rix) {
		return _mat->rows;
	}
	
	//ToDo sparse accessors
	__device__ uint32_t len_sparse() {
		return _mat->nnz;
	}
	
	__device__ uint32_t pos_sparse(uint32_t rix) {
		return _mat->row_ptr[rix];
	}
	
	__device__ uint32_t* cols_sparse(uint32_t rix) {
		return &_mat->col_idx[_mat->row_ptr[rix]];
	}
	
	__device__ T& val_sparse_rc(uint32_t r, uint32_t c) {
//		printf("TBI: val_sparse_rc\n");
//		asm("trap;");

		return _mat->data[0];
	}
	
	__device__ T& val_sparse_i(uint32_t i) {
		return _mat->data[i];
	}
	
	__device__ T* vals_sparse(uint32_t rix) {
		return &_mat->data[_mat->row_ptr[rix]];
	}
	
	__device__ uint32_t row_len_sparse(uint32_t rix) {
		return _mat->row_ptr[rix+1]-_mat->row_ptr[rix];
	}
	
	__device__ void set_sparse(uint32_t idx, uint32_t c, T v) {
//		uint32_t idx = _mat->cols*r+c;
		_mat->data[idx] = v;
		_mat->col_idx[idx] = c;
//		_mat->row_ptr[r+1] = _mat->row_ptr[r+1] > 0 ? min(idx, _mat->row_ptr[r+1]) : idx;
//		if(threadIdx.x == 0)
//		atomicMax(&(_mat->row_ptr[r+1]), idx < _mat->nnz-1 ? idx+1 : idx);
//		v == 0.0 ? atomicAdd(&(_mat->nnz), -1) : atomicAdd(&(_mat->nnz), -1);
		
//		if(blockIdx.x == 0 && threadIdx.x > 20 && threadIdx.x < 30)
//			printf("nnz=%d idx=%d r=%d c=%d v=%4.3f\n",  _mat->nnz, idx, r, c, v);
//		_mat->row_ptr[r+1] = _mat->row_ptr[r+1] > 0 ? max(idx, _mat->row_ptr[r+1]) : idx;
	}
};
#endif


#ifdef __CUDACC_RTC__

//ToDo: move to separate file
template <typename T>
struct Vector {
	T* data;
	uint32_t length;

	__device__ T* vals(uint32_t idx) { return &data[idx]; }

	__device__ T& operator[](uint32_t idx) {
		return data[idx];
	}
	
	__device__ void print(const char* name, uint32_t end_ = 0, uint32_t start = 0, uint32_t bID = 0, uint32_t tID = 0) {
		if(blockIdx.x == bID && threadIdx.x==tID) {
			uint32_t end = end_;
			if(end > 0)
				end = min(end, length);
			printf("%s: ", name);
			for(auto i = start; i < end; ++i)
				print("%4.3f ", data[i]);
		}
	}
};

template <typename T, uint32_t ELEMENTS>
class RingBuffer {
	Vector<T> vec[ELEMENTS];
	int32_t pos;

public:
	__device__ void init(uint32_t offset, uint32_t length, T* buffer) {
		pos = -1;
		for(auto i=0;i<ELEMENTS;++i) {
			vec[i].data = &buffer[offset + length * i];
			vec[i].length = length;
		}
	}

	__device__ Vector<T>& next() {
		pos = (pos+1>=ELEMENTS) ? 0 : pos+1;
		__syncthreads();
		return vec[pos];
	}
};

template <typename T, int NUM_B>
struct SpoofOp {
	MatrixAccessor<T> a;
	MatrixAccessor<T> b[NUM_B];
	MatrixAccessor<T> c;
	T* scalars;
	uint32_t grix;
	T* avals;
	uint32_t* aix;
	uint32_t alen;
	
	SpoofOp(Matrix<T>* A, Matrix<T>* B, Matrix<T>* C, T* scalars, T* tmp_stor, uint32_t grix) :
			scalars(scalars), grix(grix), avals(A->data), aix(A->col_idx) {
		a.init(A);
		c.init(C);
		alen = a.row_len(grix);

		if(B)
			for(auto i = 0; i < NUM_B; ++i)
				b[i].init(&(B[i]));
	}
	
//	__device__ Vector<T>& getTempStorage(uint32_t len) {
//		Vector<T>& vec = temp_rb.next();
//		tvec.length = len;
//		return vec;
//	}
};
#endif // __CUDACC_RTC__

#endif //SYSTEMDS_MATRIX_H
