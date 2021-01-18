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

template <typename T>
struct SpoofOp {
	__device__ virtual Vector<T>& getTempStorage(uint32_t len) = 0;
};
#endif // __CUDACC_RTC__

template <typename T>
struct Matrix {
	T* data;
	uint32_t* row_ptr;
	uint32_t* col_idx;

	uint32_t rows;
	uint32_t cols;
	uint32_t nnz;
};

template<typename T>
class MatrixAccessor {
	
	Matrix<T>* _mat;
	
	// Member function pointers
	uint32_t (MatrixAccessor::*_len)();
	uint32_t (MatrixAccessor::*_row_len)(uint32_t);
	uint32_t (MatrixAccessor::*_pos)(uint32_t);
	uint32_t* (MatrixAccessor::*_col_idxs)(uint32_t);
	
//	T (MatrixAccessor::*_val)(uint32_t, uint32_t);
	T (MatrixAccessor::*_val_r)(uint32_t);
	T (MatrixAccessor::*_val_rc)(uint32_t, uint32_t);
	T* (MatrixAccessor::*_vals)(uint32_t);

public:
	MatrixAccessor() = default;
	MatrixAccessor(Matrix<T>* mat) { init(mat); }
	void init(Matrix<T>* mat) {
		_mat = mat;
		
		if (_mat->row_ptr == nullptr) {
			_len = &MatrixAccessor::len_dense;
			_pos = &MatrixAccessor::pos_dense;
			_col_idxs = &MatrixAccessor::cols_dense;
			_val_rc = &MatrixAccessor::val_dense_rc;
			_val_r = &MatrixAccessor::val_dense_r;
			_vals = &MatrixAccessor::vals_dense_row;
			_row_len = &MatrixAccessor::row_len_dense;
		} else {
			_len = &MatrixAccessor::len_sparse;
			_pos = &MatrixAccessor::pos_sparse;
			_col_idxs = &MatrixAccessor::cols_sparse;
			_val_rc = &MatrixAccessor::val_sparse;
			_vals = &MatrixAccessor::val_sparse_row;
			_row_len = &MatrixAccessor::row_len_sparse;
		}
	}

	uint32_t cols() { return _mat->cols; }
	uint32_t rows() { return _mat->rows; }
	
	uint32_t len() { return (this->*_len)(); }
	
	uint32_t pos(uint32_t rix) {
		return (this->*_pos)(rix);
	}
	
	T val(uint32_t r, uint32_t c) {
		return (this->*_val_rc)(r, c);
	}
	
	T val(uint32_t rix) {
		return (this->*_val_r)(rix);
	}

	T* vals(uint32_t rix) {
		return (this->*_vals)(rix);
	}

	//ToDo: no sparse
    T& operator[](uint32_t i) {
	    return _mat->data[i];
    }
    
    uint32_t row_len(uint32_t rix) {
		return (this->*_row_len)(rix);
	}

	uint32_t* col_idxs(uint32_t rix) {
		return (this->*_col_idxs)(rix);
	}

private:
	uint32_t len_dense() {
		// ToDo: fix in SideInput upload
//		return _mat->cols < 2 ? _mat->rows : _mat->cols;
//		return _mat->cols;
		return _mat->rows * _mat->cols;
	}
	
	uint32_t pos_dense(uint32_t rix) {
		return _mat->cols * rix;
	}
	
	uint32_t* cols_dense(uint32_t rix) {
		printf("ERROR: no column indices array in a dense matrix! This will likely crash :-/\n");
		return nullptr;
	}
	
	T val_dense_rc(uint32_t r, uint32_t c) {
//#ifdef __CUDACC_RTC__
//		if(threadIdx.x == 0)
//		printf("bid=%d, rows=%d, cols=%d [%d,%d]=%f\n", blockIdx.x, _mat->rows, _mat->cols,
//		 		r, c, _mat->data[_mat->cols * r + c]);
//#endif
		return _mat->data[_mat->cols * r + c];
	}
	
	T val_dense_r(uint32_t rix) {
//		return &(_mat->data[_mat->cols * rix]);
		return _mat->data[rix];
	}

	// ToDo: taken over from DenseBlockFP64 - doesn't feel right though
	T* vals_dense_row(uint32_t rix) {
		return &(_mat->data[rix]);
//		return &(_mat->data[0]);
	}
	
	uint32_t row_len_dense(uint32_t rix) {
		return _mat->rows;
	}
	
	//ToDo sparse accessors
	uint32_t len_sparse() {
		return 0;
	}
	
	uint32_t pos_sparse(uint32_t rix) {
		return _mat->row_ptr[rix];
	}
	
	uint32_t* cols_sparse(uint32_t rix) {
		return &_mat->col_idx[_mat->row_ptr[rix]];
	}
	
	T val_sparse(uint32_t r, uint32_t c) {
		return _mat->data[0];
	}
	
	T* val_sparse_row(uint32_t rix) {
//		return &(_mat->data[0]);
		return &_mat->data[_mat->row_ptr[rix]];
	}
	
	uint32_t row_len_sparse(uint32_t rix) {
		return _mat->row_ptr[rix+1]-_mat->row_ptr[rix];
	}
};

#endif //SYSTEMDS_MATRIX_H
