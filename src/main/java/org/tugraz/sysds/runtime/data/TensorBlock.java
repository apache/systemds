/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.data;

import java.io.Serializable;

import org.apache.commons.lang.NotImplementedException;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public class TensorBlock implements Serializable
{
	private static final long serialVersionUID = -4205257127878517048L;
	
	public static final double SPARSITY_TURN_POINT = 0.4;
	public static final ValueType DEFAULT_VTYPE = ValueType.FP64;
	public static final int[] DEFAULT_DIMS = new int[]{0, 0};
	public static final SparseBlock.Type DEFAULT_SPARSEBLOCK = SparseBlock.Type.MCSR;
	
	//constant value type of tensor block
	protected final ValueType _vt;
	
	//min 2 dimensions to preserve proper matrix semantics
	protected int[] _dims; //[2,inf)
	protected boolean _sparse = true;
	protected long _nnz = 0;
	
	//matrix data (sparse or dense)
	protected DenseBlock _denseBlock = null;
	protected SparseBlock _sparseBlock = null;
	
	public TensorBlock() {
		this(DEFAULT_VTYPE, DEFAULT_DIMS.clone(), true, -1);
	}
	
	public TensorBlock(ValueType vt, int[] dims) {
		this(vt, dims, true, -1);
	}
	
	public TensorBlock(ValueType vt, int[] dims, boolean sp) {
		this(vt, dims, sp, -1);
	}
	
	public TensorBlock(ValueType vt, int[] dims, boolean sp, long estnnz) {
		_vt = vt;
		reset(dims, sp, estnnz, 0);
	}
	
	public TensorBlock(TensorBlock that) {
		_vt = that.getValueType();
		copy(that);
	}

	public TensorBlock(double val) {
		_vt = DEFAULT_VTYPE;
		reset(new int[] {1, 1}, false, 1, val);
	}
	
	public TensorBlock(int[] dims, ValueType vt, double val) {
		_vt = DEFAULT_VTYPE;
		_dims = dims;
		reset(dims, false, (val==0) ? 0 : getLength(), val);
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
			_denseBlock.reset(getDim(0), getDim(1), val);
		else {
			if( val != 0 ) {
				allocateDenseBlock(false);
				_denseBlock.set(val);
			} else {
				allocateDenseBlock(true);
			}
		}
	}
	
	public boolean isAllocated() {
		return _sparse ? (_sparseBlock!=null) : (_denseBlock!=null);
	}

	public TensorBlock allocateDenseBlock() {
		allocateDenseBlock(true);
		return this;
	}

	public TensorBlock allocateBlock() {
		if( _sparse )
			allocateSparseBlock();
		else
			allocateDenseBlock();
		return this;
	}
	
	public boolean allocateDenseBlock(boolean clearNNZ) {
		//allocate block if non-existing or too small (guaranteed to be 0-initialized),
        // ToDo: use reset instead, since LDRB need to check dimensions for actually available space
		long limit = getLength();
		boolean reset = (_denseBlock == null || _denseBlock.capacity() < limit);
		if( _denseBlock == null )
			// ToDo: dimensions > 2
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
	
	public int getNumDims() {
		return _dims.length;
	}
	
	public int getNumRows() {
		return getDim(0);
	}
	
	public int getNumCols() {
		return getDim(1);
	}
	
	public int getDim(int i) {
		return _dims[i];
	}
	
	public long getNonZeros() {
		return _nnz;
	}
	
	public boolean isVector() {
		return getNumDims() <= 2 
			&& (getDim(0) == 1 || getDim(1) == 1);
	}
	
	public boolean isMatrix() {
		return getNumDims() == 2 
			&& (getDim(0) > 1 && getDim(1) > 1);
	}
	
	public long getLength() {
		return UtilFunctions.prod(_dims);
	}
	
	public boolean isSparse() {
		return _sparse;
	}
	
	public boolean isEmpty() {
		return isEmpty(false);
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
				//TODO recomputeNonZeros();
			ret = (_nnz==0);
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
	
	public double get(int[] ix) {
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
			//return _sparseBlock.get(ix);
		} else {
			return _denseBlock.get(ix);
		}
	}
	
	public void set(int[] ix, double v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(ix, v);
		}
	}
	
	private void copy(TensorBlock that) {
		_dims = that._dims.clone();
		_sparse = that._sparse;
		allocateBlock();
		_nnz = that._nnz;
		
		
		// TODO Auto-generated method stub copy
	}
	
	////////
	// Size estimation and format decisions
	
	
	private boolean evalSparseFormatInMemory(int[] dims, long estnnz) {
		// TODO Auto-generated method stub
		return false;
	}
	
}
