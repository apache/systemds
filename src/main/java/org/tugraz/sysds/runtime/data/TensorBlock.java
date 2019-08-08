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

import org.apache.commons.lang.NotImplementedException;
import org.tugraz.sysds.common.Types.BlockType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.functionobjects.KahanPlus;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.functionobjects.ReduceAll;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Array;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

public class TensorBlock implements CacheBlock, Externalizable
{
	private static final long serialVersionUID = -1887367304030494999L;

	public static final double SPARSITY_TURN_POINT = 0.4;
	public static final ValueType DEFAULT_VTYPE = ValueType.FP64;
	public static final int[] DEFAULT_DIMS = new int[]{0, 0};
	public static final SparseBlock.Type DEFAULT_SPARSEBLOCK = SparseBlock.Type.MCSR;

	//constant value type of tensor block
	protected ValueType _vt;
	
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
		_vt = vt;
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
		// TODO: use reset instead, since LDRB need to check dimensions for actually available space
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

	public int[] getDims() {
		int[] dims = new int[_dims.length];
		Array.copy(_dims, 0, dims, 0, _dims.length);
		return dims;
	}

	public int getNumDims() {
		return _dims.length;
	}
	
	public int getNumRows() {
		return getDim(0);
	}

	@Override
	public int getNumColumns() {
		return getDim(1);
	}
	
	public int getDim(int i) {
		return _dims[i];
	}
	
	public long getNonZeros() {
		return _nnz;
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be outside of range.
	 *
	 * @param ix the index array which will be incremented to the next index array
	 */
	public void getNextIndexes(int[] ix) {
		int i = ix.length - 1;
		ix[i]++;
		//calculating next index
		if (ix[i] == getDim(i)) {
			while (ix[i] == getDim(i)) {
				if (i - 1 < 0) {
					//we are finished
					break;
				}
				ix[i] = 0;
				i--;
				ix[i]++;
			}
		}
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
	// Input/Output functions
	
	@Override
	@SuppressWarnings("incomplete-switch")
	public void readFields(DataInput in) 
		throws IOException 
	{
		//step 1: read header
		_vt = ValueType.values()[in.readByte()];
		_dims = new int[in.readInt()];
		for(int i=0; i<_dims.length; i++)
			_dims[i] = in.readInt();
		_nnz = in.readLong();
	
		//step 2: read tensor block data
		switch( BlockType.values()[in.readByte()] ) {
			case EMPTY_BLOCK:
				reset(_dims);
				return;
			case DENSE_BLOCK: {
				allocateDenseBlock(false);
				DenseBlock a = getDenseBlock();
				int odims = (int) UtilFunctions.prod(_dims, 1);
				for( int i=0; i<getNumRows(); i++ )
					for( int j=0; j<odims; j++ ) {
						switch( _vt ) {
							case FP32: a.set(i, j, in.readFloat()); break;
							case FP64: a.set(i, j, in.readDouble()); break;
							case INT32: a.set(i, j, in.readInt()); break;
							case INT64: a.set(i, j, in.readLong()); break;
							case BOOLEAN: a.set(i, j, in.readByte()); break;
							case STRING: a.set(new int[]{i, j}, in.readUTF()); break;
						}
					}
				break;
			}
			case SPARSE_BLOCK:
			case ULTRA_SPARSE_BLOCK:
				throw new NotImplementedException();
		}
	}
	
	@Override
	@SuppressWarnings("incomplete-switch")
	public void write(DataOutput out) 
		throws IOException 
	{
		//step 1: write header
		out.writeByte(_vt.ordinal()); // value type
		out.writeInt(getNumDims()); // num dims
		for(int i=0; i<getNumDims(); i++)
			out.writeInt(getDim(i)); // dim
		out.writeLong(getNonZeros()); // nnz
		
		//step 2: write tensor block data
		if( isEmpty(false) ) {
			//empty blocks do not need to materialize row information
			out.writeByte(BlockType.EMPTY_BLOCK.ordinal());
		}
		else if( !isSparse() ) {
			out.writeByte(BlockType.DENSE_BLOCK.ordinal());
			DenseBlock a = getDenseBlock();
			int odims = (int) UtilFunctions.prod(_dims, 1);
			for( int i=0; i<getNumRows(); i++ )
				for( int j=0; j<odims; j++ ) {
					switch( _vt ) {
						case FP32: out.writeFloat((float)a.get(i, j)); break;
						case FP64: out.writeDouble(a.get(i, j)); break;
						case INT32: out.writeInt((int)a.get(i, j)); break;
						case INT64: out.writeLong((long)a.get(i, j)); break;
						case BOOLEAN: out.writeBoolean(a.get(i, j) != 0); break;
						case STRING: out.writeUTF(a.getString(new int[] {i, j})); break;
					}
				}
		}
		else {
			throw new NotImplementedException();
		}
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
	
	public double get(int r, int c) {
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
			//return _sparseBlock.get(ix);
		} else {
			return _denseBlock.get(r, c);
		}
	}

	public long getLong(int[] ix) {
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
		} else {
			return _denseBlock.getLong(ix);
		}
	}

	public String getString(int[] ix) {
		if (_sparse) {
			// TODO: Implement sparse
			throw new NotImplementedException();
			//return _sparseBlock.get(ix);
		} else {
			return _denseBlock.getString(ix);
		}
	}

	public void set(int[] ix, double v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(ix, v);
		}
	}
	
	public void set(int r, int c, double v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(r, c, v);
		}
	}

	public void set(int[] ix, long v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(ix, v);
		}
	}

	public void set(int[] ix, String v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(ix, v);
		}
	}

	public void set(double v) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(v);
		}
	}

	public void set(String str) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			_denseBlock.set(str);
		}
	}

	public void set(TensorBlock other) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			if (other.isSparse()) {
				throw new NotImplementedException();
			} else {
				_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
			}
		}
	}

	public void set(MatrixBlock other) {
		if (_sparse) {
			throw new NotImplementedException();
		} else {
			if (other.isInSparseFormat()) {
				if (other.isEmpty()) {
					_denseBlock.set(0);
				} else {
					// TODO implement sparse set instead of converting to dense
					other.sparseToDense();
					_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
				}
			} else {
				_denseBlock.set(0, _dims[0], 0, _denseBlock.getCumODims(0), other.getDenseBlock());
			}
		}
	}

	public void copy(TensorBlock that) {
		_dims = that._dims.clone();
		_sparse = that._sparse;
		_nnz = that._nnz;
		if( that.isAllocated() ) {
			allocateBlock();
			if( !_sparse )
				copyDenseToDense(that);
			else // TODO copy sparse to dense, dense to dense or sparse to dense
				throw new NotImplementedException();
		}
	}

	public TensorBlock copyShallow(TensorBlock that) {
		_dims = that._dims.clone();
		_sparse = that._sparse;
		_nnz = that._nnz;
		if( !_sparse )
			_denseBlock = that._denseBlock;
		else
			_sparseBlock = that._sparseBlock;
		return this;
	}

	private void copyDenseToDense(TensorBlock that) {
		_nnz = that._nnz;

		//plain reset to 0 for empty input
		if( that.isEmpty(false) ) {
			if(_denseBlock!=null)
				_denseBlock.reset(that._dims);
			return;
		}

		//allocate and copy dense block
		allocateDenseBlock(false);
		_denseBlock.set(that._denseBlock);
	}

	/**
	 * Copy a part of another tensor
	 * @param lower lower index of elements to copy (inclusive)
	 * @param upper upper index of elements to copy (exclusive)
	 * @param src source tensor
	 */
	public void copy(int[] lower, int[] upper, TensorBlock src) {
		// TODO consider sparse
		if (src.isEmpty(false)) {
			return;
		}
		int rowLower = lower[0];
		int rowUpper = upper[0] + 1;
		int columnLower = lower[lower.length - 1];
		int columnUpper = upper[upper.length - 1];
		for (int i = 1; i < lower.length - 1; i++) {
			columnLower += lower[i] * _denseBlock.getCumODims(i);
			columnUpper += upper[i] * _denseBlock.getCumODims(i);
		}
		if (columnLower == columnUpper || columnUpper == 0) {
			rowUpper--;
			columnUpper = _denseBlock.getCumODims(0);
		}
		_denseBlock.set(rowLower, rowUpper, columnLower, columnUpper, src.getDenseBlock());
	}

	////////
	// Size estimation and format decisions
	
	
	private boolean evalSparseFormatInMemory(int[] dims, long estnnz) {
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
	public TensorBlock aggregateUnaryOperations(AggregateUnaryOperator op, TensorBlock result) {
		// TODO allow to aggregate along a dimension?
		// TODO performance
		if (op.aggOp.increOp.fn instanceof KahanPlus) {
			op = new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject()), op.indexFn, op.getNumThreads());
		}
		int dim0 = 1;
		int dim1 = 1;
		if (op.aggOp.correctionExists) {
			dim1 = 2;
		}
		//prepare result matrix block
		if(result==null || result._vt != _vt)
			result=new TensorBlock(_vt, new int[]{dim0, dim1}, false);
		else
			result.reset(new int[]{dim0, dim1}, false);

		if( LibTensorAgg.isSupportedUnaryAggregateOperator(op) )
			if (op.indexFn instanceof ReduceAll)
				LibTensorAgg.aggregateUnaryTensor(this, result, op);
			else
				throw new DMLRuntimeException("Only ReduceAll UnaryAggregationOperators are supported for tensor");
		else
			throw new DMLRuntimeException("Current UnaryAggregationOperator not supported for tensor");
		return result;
	}

	public void incrementalAggregate(AggregateOperator aggOp, TensorBlock partialResult) {
		if(!aggOp.correctionExists) {
			if(aggOp.increOp.fn instanceof Plus) {
				LibTensorAgg.aggregateBinaryTensor(partialResult, this, aggOp);
			}
		}
		else
			throw new DMLRuntimeException("Correction not supported. correctionLocation: "+aggOp.correctionLocation);
	}

	@Override
	public long getInMemorySize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getExactSerializedSize() {
		//header size (vt, num dims, dims, nnz, type)
		long size = 4 * (1+_dims.length) + 8 + 2;
		//serialized representation
		if( !isSparse() ) {
			switch( _vt ) {
				case INT32:
				case FP32:
					size += 4 * getLength(); break;
				case INT64:
				case FP64: 
					size += 8 * getLength(); break;
				case BOOLEAN:
					//TODO perf bits instead of bytes
					size += getLength(); break;
					//size += Math.ceil((double)getLength() / 64); break;
				case STRING:
				case UNKNOWN:
					throw new NotImplementedException();
			}
		}
		else {
			throw new NotImplementedException();
		}
		return size;
	}

	@Override
	public boolean isShallowSerialize() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		return !isSparse();
	}

	@Override
	public void toShallowSerializeBlock() {
		// TODO Auto-generated method stub

	}

	@Override
	public void compactEmptyBlock() {
		// TODO Auto-generated method stub

	}

	@Override
	public CacheBlock slice(int rl, int ru, int cl, int cu, CacheBlock block) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * Slice the current block and write into the outBlock. The offsets determines where the slice starts,
	 * the length of the blocks is given by the outBlock dimensions.
	 * @param offsets offsets where the slice starts
	 * @param outBlock sliced result block
	 * @return the sliced result block
	 */
	public TensorBlock slice(int[] offsets, TensorBlock outBlock) {
		// TODO perf
		int[] srcIx = new int[offsets.length];
		Array.copy(offsets, 0, srcIx, 0, offsets.length);
		int[] destIx = new int[offsets.length];
		for (int l = 0; l < outBlock.getLength(); l++) {
			outBlock.set(destIx, get(srcIx));
			int i = outBlock.getNumDims() - 1;
			destIx[i]++;
			//calculating next index
			while (destIx[i] == outBlock.getDim(i)) {
				destIx[i] = 0;
				srcIx[i] = offsets[i];
				i--;
				if (i < 0) {
					//we are finished
					return outBlock;
				}
				destIx[i]++;
				srcIx[i]++;
			}
		}
		return outBlock;
	}

	@Override
	public void merge(CacheBlock that, boolean appendOnly) {
		// TODO Auto-generated method stub

	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		readFields(in);
	}
}
