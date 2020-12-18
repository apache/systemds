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
	uint32_t (MatrixAccessor::*_pos)(uint32_t);
	T (MatrixAccessor::*_val)(uint32_t, uint32_t);
	T* (MatrixAccessor::*_val_r)(uint32_t);

public:
	void init(Matrix<T>* mat) {
		_mat = mat;
		if (_mat->row_ptr == nullptr) {
			_len = &MatrixAccessor::len_dense;
			_pos = &MatrixAccessor::pos_dense;
			_val = &MatrixAccessor::val_dense;
			_val_r = &MatrixAccessor::val_dense_row;
		} else {
			_len = &MatrixAccessor::len_sparse;
			_pos = &MatrixAccessor::pos_sparse;
			_val = &MatrixAccessor::val_sparse;
			_val_r = &MatrixAccessor::val_sparse_row;
		}
	}
	
	uint32_t len() {
		return (this->*_len)();
	}
	
	uint32_t pos(uint32_t rix) {
		return (this->*_pos)(rix);
	}
	
	T val(uint32_t r, uint32_t c) {
		return (this->*_val)(r, c);
	}
	
	T* val(uint32_t rix) {
		return (this->*_val_r)(rix);
	}
	
private:
	uint32_t len_dense() {
		// ToDo: fix in SideInput upload
		return _mat->cols == 1 ? _mat->rows : _mat->cols;
//		return _mat->cols;
	}
	
	uint32_t pos_dense(uint32_t rix) {
		return _mat->cols * rix;
	}
	
	T val_dense(uint32_t r, uint32_t c) {
		return _mat->data[_mat->cols * r + c];
	}
	
	T* val_dense_row(uint32_t rix) {
//		return &(_mat->data[_mat->cols * rix]);
		return &(_mat->data[rix]);
	}
	
	//ToDo sparse accessors
	uint32_t len_sparse() {
		return 0;
	}
	
	uint32_t pos_sparse(uint32_t rix) {
		return 0;
	}
	
	T val_sparse(uint32_t r, uint32_t c) {
		return _mat->data[0];
	}
	
	T* val_sparse_row(uint32_t rix) {
		return &(_mat->data[0]);
	}
};

#endif //SYSTEMDS_MATRIX_H
