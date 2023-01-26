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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.BlockType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * A <code>TensorBlock</code> is the most top level representation of a tensor. There are two types of data representation
 * which can be used: Basic/Homogeneous and Data/Heterogeneous
 * Basic supports only one <code>ValueType</code>, while Data supports multiple <code>ValueType</code>s along the column
 * axis.
 * The format determines if the <code>TensorBlock</code> uses a <code>BasicTensorBlock</code> or a <code>DataTensorBlock</code>
 * for storing the data.
 */
public class TensorBlock implements CacheBlock<TensorBlock>, Externalizable {
	private static final long serialVersionUID = -8768054067319422277L;
	
	private enum SERIALIZED_TYPES {
		EMPTY, BASIC, DATA
	}
	
	public static final int[] DEFAULT_DIMS = new int[]{0, 0};
	public static final ValueType DEFAULT_VTYPE = ValueType.FP64;

	private int[] _dims;
	private boolean _basic = true;

	private DataTensorBlock _dataTensor = null;
	private BasicTensorBlock _basicTensor = null;
	
	/**
	 * Create a <code>TensorBlock</code> with [0,0] dimension and homogeneous representation (aka. basic).
	 */
	public TensorBlock() {
		this(DEFAULT_DIMS, true);
	}
	
	/**
	 * Create a <code>TensorBlock</code> with the given dimensions and the given data representation (basic/data).
	 * @param dims dimensions
	 * @param basic if true then basic <code>TensorBlock</code> else a data type of <code>TensorBlock</code>.
	 */
	public TensorBlock(int[] dims, boolean basic) {
		_dims = dims;
		_basic = basic;
	}
	
	/**
	 * Create a basic <code>TensorBlock</code> with the given <code>ValueType</code> and the given dimensions.
	 * @param vt value type
	 * @param dims dimensions
	 */
	public TensorBlock(ValueType vt, int[] dims) {
		this(dims, true);
		_basicTensor = new BasicTensorBlock(vt, dims, false);
	}
	
	/**
	 * Create a data <code>TensorBlock</code> with the given schema and the given dimensions.
	 * @param schema schema of the columns
	 * @param dims dimensions
	 */
	public TensorBlock(ValueType[] schema, int[] dims) {
		this(dims, false);
		_dataTensor = new DataTensorBlock(schema, dims);
	}
	
	/**
	 * Create a [1,1] basic FP64 <code>TensorBlock</code> containing the given value.
	 * @param value value to put inside
	 */
	public TensorBlock(double value) {
		_dims = new int[]{1, 1};
		_basicTensor = new BasicTensorBlock(value);
	}
	
	/**
	 * Wrap the given <code>BasicTensorBlock</code> inside a <code>TensorBlock</code>.
	 * @param basicTensor basic tensor block
	 */
	public TensorBlock(BasicTensorBlock basicTensor) {
		this(basicTensor._dims, true);
		_basicTensor = basicTensor;
	}
	
	/**
	 * Wrap the given <code>DataTensorBlock</code> inside a <code>TensorBlock</code>.
	 * @param dataTensor basic tensor block
	 */
	public TensorBlock(DataTensorBlock dataTensor) {
		this(dataTensor._dims, false);
		_dataTensor = dataTensor;
	}
	
	/**
	 * Copy constructor
	 * @param that <code>TensorBlock</code> to copy
	 */
	public TensorBlock(TensorBlock that) {
		copy(that);
	}
	
	/**
	 * Reset all cells to 0.
	 */
	public void reset() {
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.reset();
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(DEFAULT_VTYPE, _dims);
			_dataTensor.reset();
		}
	}
	
	/**
	 * Reset data with new dimensions.
	 * @param dims new dimensions
	 */
	public void reset(int[] dims) {
		_dims = dims;
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.reset(dims);
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(DEFAULT_VTYPE, _dims);
			_dataTensor.reset(dims);
		}
	}
	
	public boolean isBasic() {
		return _basic;
	}
	
	public boolean isAllocated() {
		if (_basic)
			return _basicTensor != null && _basicTensor.isAllocated();
		else
			return _dataTensor != null && _dataTensor.isAllocated();
	}
	
	/**
	 * If data is not yet allocated, allocate.
	 * @return this <code>TensorBlock</code>
	 */
	public TensorBlock allocateBlock() {
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.allocateBlock();
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(DEFAULT_VTYPE, _dims);
			_dataTensor.allocateBlock();
		}
		return this;
	}
	
	public BasicTensorBlock getBasicTensor() {
		return _basicTensor;
	}
	
	public DataTensorBlock getDataTensor() {
		return _dataTensor;
	}
	
	/**
	 * Get the <code>ValueType</code> if this <code>TensorBlock</code> is homogeneous.
	 * @return <code>ValueType</code> if homogeneous, null otherwise
	 */
	public ValueType getValueType() {
		if (_basic)
			return _basicTensor == null ? DEFAULT_VTYPE : _basicTensor.getValueType();
		else
			return null;
	}
	
	/**
	 * Get the schema if this <code>TensorBlock</code> is heterogeneous.
	 * @return value type if heterogeneous, null otherwise
	 */
	public ValueType[] getSchema() {
		if (_basic)
			return null;
		else {
			if (_dataTensor == null) {
				//TODO perf, do not fill, instead save schema
				ValueType[] schema = new ValueType[getDim(1)];
				Arrays.fill(schema, DEFAULT_VTYPE);
				return schema;
			}
			else
				return _dataTensor.getSchema();
		}
	}

	public int getNumDims() {
		return _dims.length;
	}

	@Override
	public int getNumRows() {
		return getDim(0);
	}

	@Override
	public int getNumColumns() {
		return getDim(1);
	}

	@Override
	public DataCharacteristics getDataCharacteristics() {
		return new TensorCharacteristics(getLongDims(), -1);
	}
	
	@Override
	public long getInMemorySize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean isShallowSerialize() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		// TODO Auto-generated method stub
		return false;
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
	public final TensorBlock slice(IndexRange ixrange, TensorBlock ret) {
		return slice((int) ixrange.rowStart, (int) ixrange.rowEnd, (int) ixrange.colStart, (int) ixrange.colEnd, ret);
	}

	@Override
	public final TensorBlock slice(int rl, int ru) {
		return slice(rl, ru, 0, getNumColumns()-1, false, null);
	}

	@Override
	public final TensorBlock slice(int rl, int ru, boolean deep) {
		return slice(rl, ru, 0, getNumColumns()-1, deep, null);
	}

	@Override
	public final TensorBlock slice(int rl, int ru, int cl, int cu) {
		return slice(rl, ru, cl, cu, false, null);
	}

	@Override
	public final TensorBlock slice(int rl, int ru, int cl, int cu, TensorBlock ret) {
		return slice(rl, ru, cl, cu, false, ret);
	}

	@Override
	public final TensorBlock slice(int rl, int ru, int cl, int cu, boolean deep) {
		return slice(rl, ru, cl, cu, deep, null);
	}

	@Override
	public TensorBlock slice(int rl, int ru, int cl, int cu, boolean deep, TensorBlock block) {
		if( !(block instanceof TensorBlock) )
			throw new RuntimeException("TensorBlock.slice(int,int,int,int,CacheBlock) CacheBlock was no TensorBlock");
		TensorBlock tb = (TensorBlock) block;
		int[] dims = new int[_dims.length];
		dims[0] = ru - rl + 1;
		dims[1] = cu - cl + 1;
		System.arraycopy(_dims, 2, dims, 2, _dims.length - 2);
		tb.reset(dims);
		int[] offsets = new int[dims.length];
		offsets[0] = rl;
		offsets[1] = cl;
		return slice(offsets, tb);
	}

	@Override
	public void merge(TensorBlock that, boolean appendOnly) {
		throw new NotImplementedException();
	}

	@Override
	public double getDouble(int r, int c) {
		return get(r, c);
	}

	@Override
	public double getDoubleNaN(int r, int c) {
		return getDouble(r, c);
	}

	@Override
	public String getString(int r, int c) {
		double v = get(r, c);
		// NaN gets converted to null here since check for null is faster than string comp
		if(Double.isNaN(v))
			return null;
		return String.valueOf(v);
	}

	public int getDim(int i) {
		return _dims[i];
	}

	public int[] getDims() {
		return _dims;
	}

	public long[] getLongDims() {
		return Arrays.stream(_dims).mapToLong(i -> i).toArray();
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be the first one.
	 *
	 * @param dims the dims array for which we have to decide the next index
	 * @param ix the index array which will be incremented to the next index array
	 */
	public static void getNextIndexes(int[] dims, int[] ix) {
		int i = ix.length - 1;
		ix[i]++;
		//calculating next index
		if (ix[i] == dims[i]) {
			while (ix[i] == dims[i]) {
				ix[i] = 0;
				i--;
				if (i < 0) {
					//we are finished
					break;
				}
				ix[i]++;
			}
		}
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be the first one.
	 *
	 * @param ix the index array which will be incremented to the next index array
	 */
	public void getNextIndexes(int[] ix) {
		getNextIndexes(_dims, ix);
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

	public boolean isEmpty() {
		return isEmpty(false);
	}

	public boolean isEmpty(boolean safe) {
		if (_basic)
			return _basicTensor == null || _basicTensor.isEmpty(safe);
		else
			return _dataTensor == null || _dataTensor.isEmpty(safe);
	}

	public long getNonZeros() {
		if (!isAllocated())
			return 0;
		if (_basic)
			return _basicTensor.getNonZeros();
		else
			return _dataTensor.getNonZeros();
	}

	public Object get(int[] ix) {
		if (_basic && _basicTensor != null)
			return _basicTensor.get(ix);
		else if (_dataTensor != null)
			return _dataTensor.get(ix);
		return 0.0;
	}

	public double get(int r, int c) {
		if (_basic && _basicTensor != null)
			return _basicTensor.get(r, c);
		else if (_dataTensor != null)
			return _dataTensor.get(r, c);
		return 0.0;
	}

	public void set(Object v) {
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.set(v);
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(getSchema(), _dims);
			_dataTensor.set(v);
		}
	}

	public void set(MatrixBlock other) {
		if (_basic)
			_basicTensor.set(other);
		else
			throw new DMLRuntimeException("TensorBlock.set(MatrixBlock) is not yet implemented for heterogeneous tensors");
	}
	
	/**
	 * Set a cell to the value given as an `Object`.
	 * @param ix indexes in each dimension, starting with 0
	 * @param v value to set
	 */
	public void set(int[] ix, Object v) {
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.set(ix, v);
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(getSchema(), _dims);
			_dataTensor.set(ix, v);
		}
	}
	
	/**
	 * Set a cell in a 2-dimensional tensor.
	 * @param r row of the cell
	 * @param c column of the cell
	 * @param v value to set
	 */
	public void set(int r, int c, double v) {
		if (_basic) {
			if (_basicTensor == null)
				_basicTensor = new BasicTensorBlock(DEFAULT_VTYPE, _dims, false);
			_basicTensor.set(r, c, v);
		}
		else {
			if (_dataTensor == null)
				_dataTensor = new DataTensorBlock(getSchema(), _dims);
			_dataTensor.set(r, c, v);
		}
	}

	/**
	 * Slice the current block and write into the outBlock. The offsets determines where the slice starts,
	 * the length of the blocks is given by the outBlock dimensions.
	 *
	 * @param offsets  offsets where the slice starts
	 * @param outBlock sliced result block
	 * @return the sliced result block
	 */
	public TensorBlock slice(int[] offsets, TensorBlock outBlock) {
		// TODO perf
		int[] srcIx = offsets.clone();
		int[] destIx = new int[offsets.length];
		for (int l = 0; l < outBlock.getLength(); l++) {
			outBlock.set(destIx, get(srcIx));
			int i = outBlock.getNumDims() - 1;
			destIx[i]++;
			srcIx[i]++;
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

	public TensorBlock copy(TensorBlock src) {
		_dims = src._dims.clone();
		_basic = src._basic;
		if (_basic) {
			_dataTensor = null;
			_basicTensor = src._basicTensor == null ? null : new BasicTensorBlock(src._basicTensor);
		}
		else {
			_basicTensor = null;
			_dataTensor = src._dataTensor == null ? null : new DataTensorBlock(src._dataTensor);
		}
		return this;
	}
	
	/**
	 * Copy a part of another <code>TensorBlock</code>
	 * @param lower lower index of elements to copy (inclusive)
	 * @param upper upper index of elements to copy (exclusive)
	 * @param src source <code>TensorBlock</code>
	 * @return the shallow copy of the src <code>TensorBlock</code>
	 */
	public TensorBlock copy(int[] lower, int[] upper, TensorBlock src) {
		if (_basic) {
			if (src._basic) {
				_basicTensor.copy(lower, upper, src._basicTensor);
			}
			else {
				throw new DMLRuntimeException("Copying `DataTensor` into `BasicTensor` is not a safe operation.");
			}
		}
		else {
			if (src._basic) {
				// TODO perf
				_dataTensor.copy(lower, upper, new DataTensorBlock(src._basicTensor));
			}
			else {
				_dataTensor.copy(lower, upper, src._dataTensor);
			}
		}
		return this;
	}
	
	/**
	 * Copy a part of another <code>TensorBlock</code>. The difference to <code>copy()</code> is that
	 * this allows for exact sub-blocks instead of taking all consecutive data elements from lower to upper.
	 * @param lower lower index of elements to copy (inclusive)
	 * @param upper upper index of elements to copy (exclusive)
	 * @param src source <code>TensorBlock</code>
	 * @return the deep copy of the src <code>TensorBlock</code>
	 */
	public TensorBlock copyExact(int[] lower, int[] upper, TensorBlock src) {
		int[] destIx = lower.clone();
		int[] srcIx = new int[lower.length];
		long length = src.getLength();
		for (long l = 0; l < length; l++) {
			set(destIx, src.get(srcIx));
			int i = src.getNumDims() - 1;
			srcIx[i]++;
			destIx[i]++;
			//calculating next index
			while (srcIx[i] == src.getDim(i)) {
				srcIx[i] = 0;
				destIx[i] = lower[i];
				i--;
				if (i < 0) {
					//we are finished
					return this;
				}
				srcIx[i]++;
				destIx[i]++;
			}
		}
		return this;
	}

	// `getExactSerializedSize()`, `write(DataOutput)` and `readFields(DataInput)` have to match in their serialized
	// form definition
	@Override
	public long getExactSerializedSize() {
		// header size (_basic, _dims.length + _dims[*], type)
		long size = 1 + 4 * (1 + _dims.length) + 1;
		if (isAllocated()) {
			if (_basic) {
				size += 1 + getExactBlockDataSerializedSize(_basicTensor);
			}
			else {
				size += _dataTensor._schema.length;
				for (BasicTensorBlock bt : _dataTensor._colsdata) {
					if (bt != null)
						size += getExactBlockDataSerializedSize(bt);
				}
			}
		}
		return size;
	}
	
	/**
	 * Get the exact serialized size of a <code>BasicTensorBlock</code> if written by
	 * <code>TensorBlock.writeBlockData(DataOutput,BasicTensorBlock)</code>.
	 * @param bt <code>BasicTensorBlock</code>
	 * @return the size of the block data in serialized form
	 */
	public long getExactBlockDataSerializedSize(BasicTensorBlock bt) {
		// nnz, BlockType
		long size = 8 + 1;
		if (!bt.isSparse()) {
			switch (bt._vt) {
				case UINT4:
					size += getLength() / 2 + getLength() % 2;
				case UINT8:
					size += 1 * getLength(); break;
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
					int[] ix = new int[bt._dims.length];
					for (int i = 0; i < bt.getLength(); i++) {
						String s = (String) bt.get(ix);
						size += IOUtilFunctions.getUTFSize(s == null ? "" : s);
						getNextIndexes(bt.getDims(), ix);
					}
					break;
				case CHARACTER:
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
	public void write(DataOutput out) throws IOException {
		//step 1: write header (_basic, dims length, dims)
		out.writeBoolean(_basic);
		out.writeInt(_dims.length);
		for (int dim : _dims)
			out.writeInt(dim);

		//step 2: write block type
		//step 3: if tensor allocated write its data
		if (!isAllocated())
			out.writeByte(SERIALIZED_TYPES.EMPTY.ordinal());
		else if (_basic) {
			out.writeByte(SERIALIZED_TYPES.BASIC.ordinal());
			out.writeByte(_basicTensor.getValueType().ordinal());
			writeBlockData(out, _basicTensor);
		}
		else {
			out.writeByte(SERIALIZED_TYPES.DATA.ordinal());
			//write schema and colIndexes
			for (int i = 0; i < getDim(1); i++)
				out.writeByte(_dataTensor._schema[i].ordinal());
			for (BasicTensorBlock bt : _dataTensor._colsdata) {
				//present flag
				if (bt != null)
					writeBlockData(out, bt);
			}
		}
	}
	
	/**
	 * Write a <code>BasicTensorBlock</code>.
	 * @param out output stream
	 * @param bt source <code>BasicTensorBlock</code>
	 * @throws IOException if writing with the output stream fails
	 */
	public void writeBlockData(DataOutput out, BasicTensorBlock bt) throws IOException {
		out.writeLong(bt.getNonZeros()); // nnz
		if (bt.isEmpty(false)) {
			//empty blocks do not need to materialize row information
			out.writeByte(BlockType.EMPTY_BLOCK.ordinal());
		}
		else if (!bt.isSparse()) {
			out.writeByte(BlockType.DENSE_BLOCK.ordinal());
			DenseBlock a = bt.getDenseBlock();
			int odims = (int) UtilFunctions.prod(bt._dims, 1);
			int[] ix = new int[bt._dims.length];
			for (int i = 0; i < bt._dims[0]; i++) {
				ix[0] = i;
				for (int j = 0; j < odims; j++) {
					ix[ix.length - 1] = j;
					switch (bt._vt) {
						case FP32: out.writeFloat((float) a.get(i, j)); break;
						case FP64: out.writeDouble(a.get(i, j)); break;
						case INT32: out.writeInt((int) a.getLong(ix)); break;
						case INT64: out.writeLong(a.getLong(ix)); break;
						case BOOLEAN: out.writeBoolean(a.get(i, j) != 0); break;
						case STRING:
							String s = a.getString(ix);
							out.writeUTF(s == null ? "" : s);
							break;
						default:
							throw new DMLRuntimeException("Unsupported value type: "+bt._vt);
					}
				}
			}
		}
		else {
			throw new NotImplementedException();
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		//step 1: read header (_basic, dims length, dims)
		_basic = in.readBoolean();
		_dims = new int[in.readInt()];
		for (int i = 0; i < _dims.length; i++)
			_dims[i] = in.readInt();

		//step 2: read block type
		//step 3: if tensor allocated read its data
		switch (SERIALIZED_TYPES.values()[in.readByte()]) {
			case EMPTY:
				break;
			case BASIC:
				_basicTensor = new BasicTensorBlock(ValueType.values()[in.readByte()], _dims, false);
				readBlockData(in, _basicTensor);
				break;
			case DATA:
				//read schema and colIndexes
				ValueType[] schema = new ValueType[getDim(1)];
				for (int i = 0; i < getDim(1); i++)
					schema[i] = ValueType.values()[in.readByte()];
				_dataTensor = new DataTensorBlock(schema, _dims);
				for (int i = 0; i < _dataTensor._colsdata.length; i++) {
					//present flag
					if (_dataTensor._colsdata[i] != null)
						readBlockData(in, _dataTensor._colsdata[i]);
				}
				break;
		}
	}
	
	/**
	 * Read a <code>BasicTensorBlock</code>.
	 * @param in input stream
	 * @param bt destination <code>BasicTensorBlock</code>
	 * @throws IOException if reading with the input stream fails
	 */
	protected void readBlockData(DataInput in, BasicTensorBlock bt) throws IOException {
		bt._nnz = in.readLong();
		switch (BlockType.values()[in.readByte()]) {
			case EMPTY_BLOCK:
				reset(bt._dims);
				return;
			case DENSE_BLOCK: {
				bt.allocateDenseBlock(false);
				DenseBlock a = bt.getDenseBlock();
				int odims = (int) UtilFunctions.prod(bt._dims, 1);
				int[] ix = new int[bt._dims.length];
				for (int i = 0; i < bt._dims[0]; i++) {
					ix[0] = i;
					for (int j = 0; j < odims; j++) {
						ix[ix.length - 1] = j;
						switch (bt._vt) {
							case FP32: a.set(i, j, in.readFloat()); break;
							case FP64: a.set(i, j, in.readDouble()); break;
							case INT32: a.set(ix, in.readInt()); break;
							case INT64: a.set(ix, in.readLong()); break;
							case BOOLEAN: a.set(i, j, in.readByte()); break;
							case STRING:
								// FIXME readUTF is not supported for CacheDataInput
								a.set(ix, in.readUTF());
								break;
							default:
								throw new DMLRuntimeException("Unsupported value type: "+bt._vt);
						}
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
	public void writeExternal(ObjectOutput out) throws IOException {
		write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		readFields(in);
	}
	
	public TensorBlock binaryOperations(BinaryOperator op, TensorBlock thatValue, TensorBlock result) {
		if( !LibTensorBincell.isValidDimensionsBinary(this, thatValue) )
			throw new RuntimeException("Block sizes are not matched for binary cell operations");
		if (!_basic || !thatValue.isBasic())
			throw new RuntimeException("Binary operations on tensors only supported for BasicTensors at the moment");
		//prepare result matrix block
		ValueType vt = TensorBlock.resultValueType(getValueType(), thatValue.getValueType());
		if (result == null || result.getValueType() != vt)
			result = new TensorBlock(vt, _dims);
		else {
			result.reset(_dims);
		}
		
		LibTensorBincell.bincellOp(this, thatValue, result, op);
		
		return result;
	}
	
	public static ValueType resultValueType(ValueType in1, ValueType in2) {
		// TODO reconsider with operation types
		if (in1 == ValueType.UNKNOWN || in2 == ValueType.UNKNOWN)
			throw new DMLRuntimeException("Operations on unknown value types not possible");
		else if (in1 == ValueType.STRING || in2 == ValueType.STRING)
			return ValueType.STRING;
		else if (in1 == ValueType.FP64 || in2 == ValueType.FP64)
			return ValueType.FP64;
		else if (in1 == ValueType.FP32 || in2 == ValueType.FP32)
			return ValueType.FP32;
		else if (in1 == ValueType.INT64 || in2 == ValueType.INT64)
			return ValueType.INT64;
		else if (in1 == ValueType.INT32 || in2 == ValueType.INT32)
			return ValueType.INT32;
		else // Boolean - Boolean
			return ValueType.INT64;
	}
}
