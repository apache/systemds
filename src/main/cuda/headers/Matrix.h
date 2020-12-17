//
// Created by markd on 17.12.20.
//

#pragma once
#ifndef SYSTEMDS_MATRIX_H
#define SYSTEMDS_MATRIX_H

//template <typename T>
struct IMatrix {
//public:
    virtual ~IMatrix() = default;
    virtual double* val(uint32_t rix) const = 0;
//	virtual size_t pos(size_t rix) const = 0;
//	virtual size_t len() const = 0;

//protected:
	uint32_t rows;
	uint32_t cols;

    double* data;
	uint32_t * row_ptr;
	uint32_t * col_idx;

};

//template<typename Derived>
struct DenseMatT : public Matrix {
public:
    virtual double* val(uint32_t rix) const override {
        return &(data[rows * rix]);
    }
	
//	virtual size_t pos(size_t rix) const override {
//    	return rix * this->cols;
//    };
//
//	virtual size_t len() const override {
//		return this->rows * this->cols;
//	};
	
//    DenseMatT(size_t rows, size_t cols, T* data, T* row_ptr, T* col_idx)
//    :
//            Matrix<T>::rows(rows),
//            Matrix<T>::cols(cols),
//            Matrix<T>::data(data),
//            Matrix<T>::row_ptr(row_ptr),
//            Matrix<T>::col_idx(col_idx)
//			rows(rows),
//			cols(cols),
//			data(data),
//			row_ptr(row_ptr),
//			col_idx(col_idx)

//            {
//				this->rows = rows;
//						this->cols =cols;
//						this->data = data;
//						this->row_ptr = row_ptr;
//						this->col_idx = col_idx;
//            }
protected:
    DenseMatT() = default;
    DenseMatT(const DenseMatT&) = default;
    DenseMatT(DenseMatT&&) = default;
};

struct DenseMatrix : public DenseMatT {
//template<typename T>
//class DenseMatrix : public DenseMatT<DenseMatrix> {
//public:
//	DenseMatrix(size_t rows, size_t cols, T* data, T* row_ptr, T* col_idx) :
//	DenseMatrix(uint32_t rows_, uint32_t cols, double* data, double* row_ptr, double* col_idx) : Matrix{rows, cols, data, row_ptr, col_idx} {
//			DenseMatT<T, DenseMatrix<T>>(rows, cols, data, row_ptr, col_idx) {};
//			rows(rows_), cols(cols), data(data), row_ptr(row_ptr), col_idx(col_idx) {
	DenseMatrix(uint32_t rows_, uint cols_, double* data_, uint32_t* row_ptr_, uint32_t* col_idx_) {
		rows = rows_;
		cols = cols_;
		data = data_;
		row_ptr = row_ptr_;
		col_idx = col_idx_;
	}
};

//class DenseMatrixFP32 : public DenseMatT<float, DenseMatrixFP32> {};
//class DenseMatrixFP64 : public DenseMatT<double, DenseMatrixFP64> {};

#endif //SYSTEMDS_MATRIX_H
