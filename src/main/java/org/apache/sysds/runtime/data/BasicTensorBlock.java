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

package org.apache.sysds.runtime.data;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.Serializable;

import static org.apache.sysds.runtime.data.LibTensorAgg.aggregateBinaryTensor;
import static org.apache.sysds.runtime.data.TensorBlock.DEFAULT_DIMS;
import static org.apache.sysds.runtime.data.TensorBlock.DEFAULT_VTYPE;

public class BasicTensorBlock implements Serializable {
	private static final long serialVersionUID = -7665685894181661833L;

	public static final double SPARSITY_TURN_POINT = 0.4;

	public static final SparseBlock.Type DEFAULT_SPARSEBLOCK = SparseBlock.Type.MCSR;

	protected int[] _dims;
	//constant value type of tensor block
	protected ValueType _vt;

	protected boolean _sparse = true;
	protected long _nnz = 0;

	//matrix data (sparse or dense)
	protected DenseBlock _denseBlock = null;
	protected SparseBlock _sparseBlock = null;

	public BasicTensorBlock() {
		this(DEFAULT_VTYPE, DEFAULT_DIMS.clone(), true, -1);
	}

	public BasicTensorBlock(ValueType vt, int[] dims) {
		this(vt, dims, true, -1);
	}

	public BasicTensorBlock(ValueType vt, int[] dims, boolean sp) {
		this(vt, dims, sp, -1);
	}

	public BasicTensorBlock(ValueType vt, int[] dims, boolean sp, long estnnz) {
		_vt = vt;
		reset(dims, sp, estnnz, 0);
	}

	public BasicTensorBlock(BasicTensorBlock that) {
		_vt = that.getValueType();
		copy(that);
	}

	public BasicTensorBlock(double val) {
		_vt = DEFAULT_VTYPE;
		reset(new int[] {1, 1}, false, 1, val);
	}

	public BasicTensorBlock(int[] dims, ValueType vt, double val) {
		_vt = vt;
		_dims = dims;
		reset(dims, false, (val==0) ? 0 : getLength(), val);
	}

	public long getLength() {
		return UtilFunctions.prod(_dims);
	}

	////////
	// Initialization methods
	// (reset, init, allocate, etc)

	public void reset() {
		reset(_dims, _sparse, -1, 0);
	}

	public void reset(int[] dims) {
		reset(dims, _sparse, -1, 0);
	}

	public void reset(int[] dims, long estnnz) {
		reset(dims, evalSparseFormatInMemory(dims, estnnz), estnnz, 0);
	}

	public void reset(int[] dims, boolean sp) {
		reset(dims, sp, -1, 0);
	}

	public void reset(int[] dims, boolean sp, long estnnz) {
		reset(dims, sp, estnnz, 0);
	}

	/**
	 * Internal canonical reset of dense and sparse tensor blocks.
	 *
	 * @param dims    number and size of dimensions
	 * @param sp      sparse representation
	 * @param estnnz  estimated number of non-zeros
	 * @param val     initialization value
	 */
	private void reset(int[] dims, boolean sp, long estnnz, double val) {
		//check for valid dimensions
		if( dims.length < 2 )
			throw new DMLRuntimeException("Invalid number of tensor dimensions: " + dims.length);
		for( int i=0; i<dims.length; i++ )
			if( dims[i] < 0 )
				throw new DMLRuntimeException("Invalid "+i+"th dimensions: "+dims[i]);

		//reset basic meta data
		_dims = dims;
		_sparse = sp;
		_nnz = (val == 0) ? 0 : getLength();

		//reset sparse/dense blocks
		if( _sparse )
			resetSparse();
		else
			resetDense(val);
	}

	private void resetSparse() {
		if(_sparseBlock == null)
			return;
		//TODO simplify estimated non-zeros
		_sparseBlock.reset(-1, getDim(2));
	}

	private void resetDense(double val) {
		//handle to dense block allocation and
		//reset dense block to given value
		if( _denseBlock != null )
			_denseBlock.reset(_dims, val);
		else {
			if( val != 0 ) {
				allocateDenseBlock(false);
				_denseBlock.set(val);
			}
			else {
				allocateDenseBlock(true);
			}
		}
	}
	
	/**
	 * Recomputes and materializes the number of non-zero values
	 * of the entire basic tensor block.
	 *
	 * @return number of non-zeros
	 */
	public long recomputeNonZeros() {
		if( _sparse && _sparseBlock != null ) { //SPARSE
			throw new DMLRuntimeException("Sparse tensor block not supported");
		}
		else if( !_sparse && _denseBlock != null ) { //DENSE
			_nnz = _denseBlock.countNonZeros();
		}
		return _nnz;
	}

	public boolean isAllocated() {
		return _sparse ? (_sparseBlock != null) : (_denseBlock != null);
	}

	public BasicTensorBlock allocateDenseBlock() {
		allocateDenseBlock(true);
		return this;
	}

	public BasicTensorBlock allocateBlock() {
		if( _sparse )
			allocateSparseBlock();
		else
			allocateDenseBlock();
		return this;
	}

	public boolean allocateDenseBlock(boolean clearNNZ) {
		//allocate block if non-existing or too small (guaranteed to be 0-initialized),
		// TODO: use _denseBlock.reset instead, since LDRB need to check dimensions for actually available space
		long limit = getLength();
		boolean reset = (_denseBlock == null || _denseBlock.capacity() < limit);
		if( _denseBlock == null )
			_denseBlock = DenseBlockFactory.createDenseBlock(_vt, _dims);
		else if( _denseBlock.capacity() < limit )
			_denseBlock.reset(_dims);

		//clear nnz if necessary
		if( clearNNZ )
			_nnz = 0;
		_sparse = false;

		return reset;
	}

	public boolean allocateSparseBlock() {
		return allocateSparseBlock(true);
	}

	public boolean allocateSparseBlock(boolean clearNNZ) {
		//allocate block if non-existing or too small (guaranteed to be 0-initialized)
		//but do not replace existing block even if not in default type
		boolean reset = _sparseBlock == null || _sparseBlock.numRows()<getDim(0);
		if( reset ) {
			_sparseBlock = SparseBlockFactory
					.createSparseBlock(DEFAULT_SPARSEBLOCK, getDim(0));
		}
		//clear nnz if necessary
		if( clearNNZ )
			_nnz = 0;

		return reset;
	}

	////////
	// Basic meta data

	public ValueType getValueType() {
		return _vt;
	}

	public long getNonZeros() {
		return _nnz;
	}

	public int getNumRows() {
		return getDim(0);
	}

	public int getNumColumns() {
		return getDim(1);
	}

	public int getNumDims() {
		return _dims.length;
	}

	public int getDim(int i) {
		return _dims[i];
	}

	public int[] getDims() {
		return _dims;
	}

	public boolean isSparse() {
		return _sparse;
	}

	public boolean isEmpty(boolean safe) {
		boolean ret = false;
		if( _sparse && _sparseBlock==null )
			ret = true;
		else if( !_sparse && _denseBlock==null )
			ret = true;
		if( _nnz==0 ) {
			//prevent under-estimation
			if(safe)
				recomputeNonZeros();
			ret = (_nnz == 0);
		}
		return ret;
	}

	public DenseBlock getDenseBlock() {
		return _denseBlock;
	}

	public SparseBlock getSparseBlock() {
		return _sparseBlock;
	}

	////////
	// Basic modification

	public Object get(int[] ix) {
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
		}
		else {
			switch (_vt) {
				case FP64:
					return _denseBlock.get(ix);
				case FP32:
					return (float) _denseBlock.get(ix);
				case INT64:
					return _denseBlock.getLong(ix);
				case INT32:
					return (int) _denseBlock.getLong(ix);
				case BOOLEAN:
					return _denseBlock.get(ix) != 0;
				case STRING:
					return _denseBlock.getString(ix);
				default:
					throw new DMLRuntimeException("Unsupported value type: " + _vt);
			}
		}
	}

	public double get(int r, int c) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("BasicTensor.get(int,int) dimension mismatch: expected=2 actual=" + getNumDims());
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
			//return _sparseBlock.get(ix);
		}
		else {
			return _denseBlock.get(r, c);
		}
	}

	public void set(int[] ix, Object v) {
		if (_sparse) {
			throw new NotImplementedException();
		}
		else if (v != null) {
			if (v instanceof Double) {
				double old = _denseBlock.get(ix);
				_denseBlock.set(ix, (Double) v);
				_nnz += (old == 0 ? 0 : -1) + ((Double) v == 0 ? 0 : 1);
			}
			else if (v instanceof Float) {
				double old = _denseBlock.get(ix);
				_denseBlock.set(ix, (Float) v);
				_nnz += (old == 0 ? 0 : -1) + ((Float) v == 0 ? 0 : 1);
			}
			else if (v instanceof Long) {
				long old = _denseBlock.getLong(ix);
				_denseBlock.set(ix, (Long) v);
				_nnz += (old == 0 ? 0 : -1) + ((Long) v == 0 ? 0 : 1);
			}
			else if (v instanceof Integer) {
				long old = _denseBlock.getLong(ix);
				_denseBlock.set(ix, (Integer) v);
				_nnz += (old == 0 ? 0 : -1) + ((Integer) v == 0 ? 0 : 1);
			}
			else if (v instanceof Boolean) {
				long old = _denseBlock.getLong(ix);
				_denseBlock.set(ix, ((Boolean) v) ? 1.0 : 0.0);
				_nnz += (old == 0 ? 0 : -1) + (!(Boolean) v ? 0 : 1);
			}
			else if (v instanceof String) {
				String old = _denseBlock.getString(ix);
				if (old != null && !old.isEmpty())
					_nnz--;
				_denseBlock.set(ix, (String) v);
				if (!((String) v).isEmpty())
					_nnz++;
			}
			else
				throw new DMLRuntimeException("BasicTensor.set(int[],Object) is not implemented for the given Object");
		}
	}

	public void set(int r, int c, double v) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("BasicTensor.set(int,int,double) dimension mismatch: expected=2 actual=" + getNumDims());
		if (_sparse) {
			throw new NotImplementedException();
		}
		else {
			double old = _denseBlock.get(r, c);
			_denseBlock.set(r, c, v);
			_nnz += (old == 0 ? 0 : -1) + (v == 0 ? 0 : 1);
		}
	}

	public void set(double v) {
		if (_sparse) {
			throw new NotImplementedException();
		}
		else {
			_denseBlock.set(v);
			if (v == 0)
				_nnz = 0;
			else
				_nnz = getLength();
		}
	}

	public void set(Object v) {
		if (_sparse) {
			throw new NotImplementedException();
		}
		else {
			if (v instanceof Double) {
				_denseBlock.set((Double) v);
				_nnz += ((Double) v == 0 ? 0 : 1);
			}
			else if (v instanceof Float) {
				_denseBlock.set((Float) v);
				_nnz += ((Float) v == 0 ? 0 : 1);
			}
			else if (v instanceof Long) {
				_denseBlock.set((Long) v);
				_nnz += ((Long) v == 0 ? 0 : 1);
			}
			else if (v instanceof Integer) {
				_denseBlock.set((Integer) v);
				_nnz += ((Integer) v == 0 ? 0 : 1);
			}
			else if (v instanceof Boolean) {
				_denseBlock.set(((Boolean) v) ? 1.0 : 0.0);
				_nnz += (!(Boolean) v ? 0 : 1);
			}
			else if (v instanceof String) {
				_denseBlock.set((String) v);
				_nnz += (((String) v).isEmpty() ? 0 : 1);
			}
			else
				throw new DMLRuntimeException("BasicTensor.set(Object) is not implemented for the given Object");
		}
	}

	public void set(BasicTensorBlock other) {
		if (_sparse)
			throw new NotImplementedException();
		else {
			if (other.isSparse())
				throw new NotImplementedException();
			else {
				_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
				_nnz = other._nnz;
			}
		}
	}

	public void set(MatrixBlock other) {
		if (_sparse) {
			throw new NotImplementedException();
		}
		else {
			if (other.isInSparseFormat()) {
				if (other.isEmpty()) {
					_denseBlock.set(0);
				}
				else {
					// TODO implement sparse set instead of converting to dense
					other.sparseToDense();
					_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
					_nnz = other.getNonZeros();
				}
			}
			else {
				_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
				_nnz = other.getNonZeros();
			}
		}
	}

	public void copy(BasicTensorBlock that) {
		_dims = that._dims.clone();
		_sparse = that._sparse;
		_nnz = that._nnz;
		if (that.isAllocated()) {
			if (!_sparse)
				copyDenseToDense(that);
			else // TODO copy sparse to dense, dense to dense or sparse to dense
				throw new NotImplementedException();
		}
	}

	public BasicTensorBlock copyShallow(BasicTensorBlock that) {
		_dims = that._dims.clone();
		_sparse = that._sparse;
		_nnz = that._nnz;
		if (!_sparse)
			_denseBlock = that._denseBlock;
		else
			_sparseBlock = that._sparseBlock;
		return this;
	}

	private void copyDenseToDense(BasicTensorBlock that) {
		_nnz = that._nnz;

		//plain reset to 0 for empty input
		if (that.isEmpty(false)) {
			if (_denseBlock != null)
				_denseBlock.reset(that._dims);
			else
				_denseBlock = DenseBlockFactory.createDenseBlock(that._vt, that._dims);
		}
		//allocate and copy dense block
		allocateDenseBlock(false);
		_denseBlock.set(that._denseBlock);
	}

	/**
	 * Copy a part of another <code>BasicTensorBlock</code>
	 * @param lower lower index of elements to copy (inclusive)
	 * @param upper upper index of elements to copy (exclusive)
	 * @param src source <code>BasicTensorBlock</code>
	 */
	public void copy(int[] lower, int[] upper, BasicTensorBlock src) {
		// TODO consider sparse
		if (src.isEmpty(false)) {
			return;
		}
		DenseBlock db = src.getDenseBlock();
		int rowLower = lower[0];
		int rowUpper = upper[0] + 1;
		int columnLower = lower[lower.length - 1];
		int columnUpper = upper[upper.length - 1];
		for (int i = 1; i < lower.length - 1; i++) {
			columnLower += lower[i] * db.getCumODims(i);
			columnUpper += upper[i] * db.getCumODims(i);
		}
		if (columnLower == columnUpper || columnUpper == 0) {
			rowUpper--;
			columnUpper = db.getCumODims(0);
		}
		_denseBlock.set(rowLower, rowUpper, columnLower, columnUpper, db);
	}

	////////
	// Size estimation and format decisions
	private static boolean evalSparseFormatInMemory(int[] dims, long estnnz) {
		// TODO Auto-generated method stub
		return false;
	}

	///////
	// Aggregations

	/**
	 * Aggregate a unary operation on this tensor.
	 * @param op the operation to apply
	 * @param result the result tensor
	 * @return the result tensor
	 */
	public BasicTensorBlock aggregateUnaryOperations(AggregateUnaryOperator op, BasicTensorBlock result) {
		// TODO allow to aggregate along a dimension?
		// TODO performance
		if (op.aggOp.increOp.fn instanceof KahanPlus) {
			op = new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject()), op.indexFn, op.getNumThreads());
		}
		int dim0 = 1;
		int dim1 = 1;
		if (op.aggOp.existsCorrection()) {
			dim1 = 2;
		}
		//prepare result matrix block
		if (result == null || result._vt != _vt)
			result = new BasicTensorBlock(_vt, new int[]{dim0, dim1}, false);
		else
			result.reset(new int[]{dim0, dim1}, false);

		if (LibTensorAgg.isSupportedUnaryAggregateOperator(op))
			if (op.indexFn instanceof ReduceAll)
				LibTensorAgg.aggregateUnaryTensor(this, result, op);
			else
				throw new DMLRuntimeException("Only ReduceAll UnaryAggregationOperators are supported for tensor");
		else
			throw new DMLRuntimeException("Current UnaryAggregationOperator not supported for tensor");
		return result;
	}

	public void incrementalAggregate(AggregateOperator aggOp, BasicTensorBlock partialResult) {
		if( !aggOp.existsCorrection() && aggOp.increOp.fn instanceof Plus)
			aggregateBinaryTensor(partialResult, this, aggOp);
		else
			throw new DMLRuntimeException("Correction not supported. correctionLocation: " + aggOp.correction);
	}
}
