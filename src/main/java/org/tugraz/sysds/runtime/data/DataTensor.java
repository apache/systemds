/*
 * Copyright 2019 Graz University of Technology
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

import org.apache.commons.lang.math.IntRange;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.BinaryOperator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

public class DataTensor extends TensorBlock {
	private static final int VALID_VALUE_TYPES_LENGTH = ValueType.values().length - 1;

	private BasicTensor[] _colsdata = new BasicTensor[VALID_VALUE_TYPES_LENGTH];
	private ValueType[] _schema = null;
	private int[] _colsToIx = null;
	private int[][] _ixToCols = null;

	public DataTensor() {
		this(new ValueType[0], DEFAULT_DIMS);
	}

	public DataTensor(int ncols, ValueType vt) {
		this(vt, new int[]{0, ncols});
	}

	public DataTensor(ValueType[] schema) {
		_dims = new int[]{0, schema.length};
		_schema = schema;
		_colsToIx = new int[_schema.length];
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		int[] typeIxCounter = new int[VALID_VALUE_TYPES_LENGTH];
		for (int i = 0; i < schema.length; i++) {
			int type = schema[i].ordinal();
			_colsToIx[i] = typeIxCounter[type]++;
		}
		for (int i = 0; i < schema.length; i++) {
			int type = schema[i].ordinal();
			if (_ixToCols[type] == null) {
				_ixToCols[type] = new int[typeIxCounter[type]];
				typeIxCounter[type] = 0;
			}
			_ixToCols[type][typeIxCounter[type]++] = i;
		}
	}

	public DataTensor(ValueType[] schema, int[] dims) {
		_dims = dims;
		_schema = schema;
		_colsToIx = new int[_schema.length];
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		int[] typeIxCounter = new int[VALID_VALUE_TYPES_LENGTH];
		for (int i = 0; i < schema.length; i++) {
			int type = schema[i].ordinal();
			_colsToIx[i] = typeIxCounter[type]++;
		}
		for (int i = 0; i < schema.length; i++) {
			int type = schema[i].ordinal();
			if (_ixToCols[type] == null) {
				_ixToCols[type] = new int[typeIxCounter[type]];
				typeIxCounter[type] = 0;
			}
			_ixToCols[type][typeIxCounter[type]++] = i;
		}
		reset();
	}

	public DataTensor(ValueType vt, int[] dims) {
		_dims = dims;
		_schema = new ValueType[getDim(1)];
		Arrays.fill(_schema, vt);
		_colsToIx = new IntRange(0, getDim(1)).toArray();
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		_ixToCols[vt.ordinal()] = new IntRange(0, getDim(1)).toArray();
		reset();
	}

	public DataTensor(ValueType[] schema, int[] dims, String[][] data) {
		this(schema, dims);
		allocateBlock();
		for (int i = 0; i < schema.length; i++) {
			int[] ix = new int[dims.length];
			ix[1] = _colsToIx[i];
			BasicTensor current = _colsdata[schema[i].ordinal()];
			for (int j = 0; j < data[i].length; j++) {
				current.set(ix, data[i][j]);
				current.getNextIndexes(ix);
				if (ix[1] != _colsToIx[i]) {
					// We want to stay in the current column
					if (ix[1] == 0)
						ix[1] = _colsToIx[i];
					else {
						ix[1] = _colsToIx[i];
						ix[0]++;
					}
				}
			}
		}
	}

	public DataTensor(double val) {
		_dims = new int[]{1, 1};
		_schema = new ValueType[]{ValueType.FP64};
		_colsToIx = new int[] {0};
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		_ixToCols[ValueType.FP64.ordinal()] = new int[] {0};
		_colsdata = new BasicTensor[VALID_VALUE_TYPES_LENGTH];
		_colsdata[ValueType.FP64.ordinal()] = new BasicTensor(val);
	}

	public DataTensor(DataTensor that) {
		copy(that);
	}

	@Override
	public void reset() {
		reset(_dims);
	}

	@Override
	public void reset(int[] dims) {
		if (dims.length < 2)
			throw new DMLRuntimeException("DataTensor.reset(int[]) invalid number of tensor dimensions: " + dims.length);
		if (dims[1] > _dims[1])
			throw new DMLRuntimeException("DataTensor.reset(int[]) columns can not be added without a provided schema," +
					" use reset(int[],ValueType[]) instead");
		for (int i = 0; i < dims.length; i++) {
			if (dims[i] < 0)
				throw new DMLRuntimeException("DataTensor.reset(int[]) invalid  dimension " + i + ": " + dims[i]);
		}
		_dims = dims;
		if (getDim(1) < _schema.length) {
			ValueType[] schema = new ValueType[getDim(1)];
			System.arraycopy(_schema, 0, schema, 0, getDim(1));
			_schema = schema;
		}
		reset(_dims, _schema);
	}

	public void reset(int[] dims, ValueType[] schema) {
		if (dims.length < 2)
			throw new DMLRuntimeException("DataTensor.reset(int[],ValueType[]) invalid number of tensor dimensions: " + dims.length);
		if (dims[1] != schema.length)
			throw new DMLRuntimeException("DataTensor.reset(int[],ValueType[]) column dimension and schema length does not match");
		for (int i = 0; i < dims.length; i++)
			if (dims[i] < 0)
				throw new DMLRuntimeException("DataTensor.reset(int[],ValueType[]) invalid  dimension " + i + ": " + dims[i]);
		_dims = dims;
		_schema = schema;
		_colsToIx = new int[_schema.length];
		int[] typeIxCounter = new int[VALID_VALUE_TYPES_LENGTH];
		for (int i = 0; i < schema.length; i++) {
			int type = schema[i].ordinal();
			_colsToIx[i] = typeIxCounter[type]++;
		}
		int[] colCounters = new int[VALID_VALUE_TYPES_LENGTH];
		for (int i = 0; i < getDim(1); i++) {
			int type = _schema[i].ordinal();
			if (_ixToCols[type] == null || _ixToCols[type].length != typeIxCounter[type]) {
				_ixToCols[type] = new int[typeIxCounter[type]];
			}
			_ixToCols[type][colCounters[type]++] = i;
		}
		// typeIxCounter now has the length of the BasicTensors
		if (_colsdata == null) {
			allocateBlock();
		} else {
			for (int i = 0; i < _colsdata.length; i++) {
				if (_colsdata[i] != null) {
					_colsdata[i].reset(toInternalDims(dims, typeIxCounter[i]));
				}
				else if (typeIxCounter[i] != 0) {
					int[] colDims = toInternalDims(_dims, typeIxCounter[i]);
					_colsdata[i] = new BasicTensor(ValueType.values()[i], colDims, false);
					_colsdata[i].allocateBlock();
				}
			}
		}
	}

	@Override
	public DataTensor allocateBlock() {
		if (_colsdata == null)
			// -1 because of unknown
			_colsdata = new BasicTensor[VALID_VALUE_TYPES_LENGTH];
		int[] colDataColumnLength = new int[_colsdata.length];
		for (ValueType valueType : _schema)
			colDataColumnLength[valueType.ordinal()]++;
		for (int i = 0; i < _colsdata.length; i++) {
			if (colDataColumnLength[i] != 0) {
				int[] dims = toInternalDims(_dims, colDataColumnLength[i]);
				// TODO sparse
				_colsdata[i] = new BasicTensor(ValueType.values()[i], dims, false);
				_colsdata[i].allocateBlock();
			}
		}
		return this;
	}

	@Override
	public boolean isAllocated() {
		if (_colsdata == null)
			return false;
		for (BasicTensor block : _colsdata) {
			if (block != null && block.isAllocated())
				return true;
		}
		return false;
	}

	@Override
	public boolean isEmpty(boolean safe) {
		if (!isAllocated()) {
			return true;
		}
		for (BasicTensor tb : _colsdata) {
			if (tb != null && !tb.isEmpty(safe))
				return false;
		}
		return true;
	}

	@Override
	public long getNonZeros() {
		// TODO non zero for DataTensor
		if (!isAllocated()) {
			return 0;
		}
		long nnz = 0;
		for (BasicTensor bt : _colsdata) {
			if (bt != null)
				nnz += bt.getNonZeros();
		}
		return nnz;
	}

	public ValueType[] getSchema() {
		return _schema;
	}

	public ValueType getColValueType(int col) {
		return _schema[col];
	}

	@Override
	public Object get(int[] ix) {
		int columns = ix[1];
		int[] internalIx = toInternalIx(ix, _colsToIx[columns]);
		return _colsdata[_schema[columns].ordinal()].get(internalIx);
	}

	@Override
	public double get(int r, int c) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("DataTensor.get(int,int) dimension mismatch: expected=2 actual=" + getNumDims());
		return _colsdata[_schema[c].ordinal()].get(r, _colsToIx[c]);
	}

	@Override
	public void set(int[] ix, Object v) {
		int columns = ix[1];
		int[] internalIx = toInternalIx(ix, _colsToIx[columns]);
		_colsdata[_schema[columns].ordinal()].set(internalIx, v);
	}

	@Override
	public void set(int r, int c, double v) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("DataTensor.set(int,int,double) dimension mismatch: expected=2 actual=" + getNumDims());
		_colsdata[_schema[c].ordinal()].set(r, _colsToIx[c], v);
	}

	@Override
	public TensorBlock aggregateUnaryOperations(AggregateUnaryOperator op, TensorBlock result) {
		// TODO aggregateUnaryOperations for DataTensor
		throw new DMLRuntimeException("DataTensor.aggregateUnaryOperations is not implemented yet.");
	}

	@Override
	public void incrementalAggregate(AggregateOperator aggOp, TensorBlock partialResult) {
		// TODO incrementalAggregate for DataTensor
		throw new DMLRuntimeException("DataTensor.incrementalAggregate is not implemented yet.");
	}

	@Override
	public TensorBlock binaryOperations(BinaryOperator op, TensorBlock thatValue, TensorBlock result) {
		// TODO binaryOperations for DataTensor
		throw new DMLRuntimeException("DataTensor.binaryOperations is not implemented yet.");
	}

	@Override
	protected DataTensor checkType(TensorBlock that) {
		if (that instanceof DataTensor)
			return (DataTensor) that;
		else
			throw new DMLRuntimeException("BasicTensor.checkType(TensorBlock) given TensorBlock was no BasicTensor");
	}

	public void copy(DataTensor that) {
		_dims = that._dims.clone();
		_schema = that._schema.clone();
		_colsToIx = that._colsToIx.clone();
		if (that.isAllocated()) {
			if (that.isEmpty(false)) {
				return;
			}
			for (int i = 0; i < _colsdata.length; i++) {
				if (that._colsdata[i] != null) {
					_colsdata[i] = new BasicTensor(that._colsdata[i]);
				}
			}
		}
	}

	@Override
	public long getInMemorySize() {
		// TODO in memory size
		return 0;
	}

	@Override
	public long getExactSerializedSize() {
		//header size (num dims, dims)
		long size = 4 * (1 + _dims.length) + getDim(1) * (1 + 4);
		//colsdata serialized representation
		for (BasicTensor bt : _colsdata) {
			size += 1; // flag
			if (bt != null)
				size += bt.getExactSerializedSize();
		}
		return size;
	}

	@Override
	public boolean isShallowSerialize() {
		// TODO is shallow serialize
		return false;
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		// TODO is shallow serialize
		return false;
	}

	@Override
	public void toShallowSerializeBlock() {
		// TODO to shallow serialize block
	}

	@Override
	public void compactEmptyBlock() {
		// TODO compact empty block
	}

	@Override
	public CacheBlock slice(int rl, int ru, int cl, int cu, CacheBlock block) {
		// TODO slice
		return null;
	}

	@Override
	public void merge(CacheBlock that, boolean appendOnly) {
		// TODO merge
	}

	@Override
	public void write(DataOutput out) throws IOException {
		//step 1: write dims
		out.writeInt(getNumDims()); // num dims
		for (int i = 0; i < getNumDims(); i++)
			out.writeInt(getDim(i)); // dim
		//step 2: write schema and colIndexes
		for (int i = 0; i < getDim(1); i++) {
			out.writeByte(_schema[i].ordinal());
			out.writeInt(_colsToIx[i]);
		}
		//step 3: write basic tensors
		for (BasicTensor colsdatum : _colsdata) {
			//present flag
			if (colsdatum == null)
				out.writeBoolean(false);
			else {
				out.writeBoolean(true);
				colsdatum.write(out);
			}
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		//step 1: read dims
		_dims = new int[in.readInt()];
		for (int i = 0; i < _dims.length; i++)
			_dims[i] = in.readInt();
		_schema = new ValueType[getDim(1)];
		_colsToIx = new int[getDim(1)];
		for (int i = 0; i < getDim(1); i++) {
			_schema[i] = ValueType.values()[in.readByte()];
			_colsToIx[i] = in.readInt();
		}
		_colsdata = new BasicTensor[VALID_VALUE_TYPES_LENGTH];
		for (int i = 0; i < _colsdata.length; i++) {
			if (in.readBoolean()) {
				_colsdata[i] = new BasicTensor();
				_colsdata[i].readFields(in);
			}
		}
	}

	private static int[] toInternalIx(int[] ix, int col) {
		int[] internalIx = ix.clone();
		internalIx[1] = col;
		return internalIx;
	}

	private static int[] toInternalDims(int[] dims, int cols) {
		int[] internalDims = dims.clone();
		internalDims[1] = cols;
		return internalDims;
	}
}
