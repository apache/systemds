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

import org.apache.commons.lang.math.IntRange;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

import static org.apache.sysds.runtime.data.TensorBlock.DEFAULT_DIMS;

public class DataTensorBlock implements Serializable {
	private static final long serialVersionUID = 3410679389807309521L;

	private static final int VALID_VALUE_TYPES_LENGTH = ValueType.values().length - 1;

	protected int[] _dims;
	protected BasicTensorBlock[] _colsdata = new BasicTensorBlock[VALID_VALUE_TYPES_LENGTH];
	protected ValueType[] _schema = null;
	/**
	 * Contains the (column) index in `_colsdata` for a certain column of the `DataTensor`. Which `_colsdata` to use is specified by the `_schema`
	 */
	protected int[] _colsToIx = null;
	/**
	 * Contains the column of `DataTensor` an `_colsdata` (column) index corresponds to.
	 */
	protected int[][] _ixToCols = null;

	public DataTensorBlock() {
		this(new ValueType[0], DEFAULT_DIMS);
	}

	public DataTensorBlock(int ncols, ValueType vt) {
		this(vt, new int[]{0, ncols});
	}

	public DataTensorBlock(ValueType[] schema) {
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

	public DataTensorBlock(ValueType[] schema, int[] dims) {
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

	public DataTensorBlock(ValueType vt, int[] dims) {
		_dims = dims;
		_schema = new ValueType[getDim(1)];
		Arrays.fill(_schema, vt);
		_colsToIx = new IntRange(0, getDim(1)).toArray();
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		_ixToCols[vt.ordinal()] = new IntRange(0, getDim(1)).toArray();
		reset();
	}

	public DataTensorBlock(ValueType[] schema, int[] dims, String[][] data) {
		this(schema, dims);
		allocateBlock();
		for (int i = 0; i < schema.length; i++) {
			int[] ix = new int[dims.length];
			ix[1] = _colsToIx[i];
			BasicTensorBlock current = _colsdata[schema[i].ordinal()];
			for (int j = 0; j < data[i].length; j++) {
				current.set(ix, data[i][j]);
				TensorBlock.getNextIndexes(_dims, ix);
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

	public DataTensorBlock(double val) {
		_dims = new int[]{1, 1};
		_schema = new ValueType[]{ValueType.FP64};
		_colsToIx = new int[]{0};
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		_ixToCols[ValueType.FP64.ordinal()] = new int[]{0};
		_colsdata = new BasicTensorBlock[VALID_VALUE_TYPES_LENGTH];
		_colsdata[ValueType.FP64.ordinal()] = new BasicTensorBlock(val);
	}

	public DataTensorBlock(DataTensorBlock that) {
		copy(that);
	}

	public DataTensorBlock(BasicTensorBlock that) {
		_dims = that._dims;
		_schema = new ValueType[_dims[1]];
		Arrays.fill(_schema, that._vt);
		_colsToIx = IntStream.range(0, _dims[1]).toArray();
		_ixToCols = new int[VALID_VALUE_TYPES_LENGTH][];
		_ixToCols[that._vt.ordinal()] = IntStream.range(0, _dims[1]).toArray();
		_colsdata = new BasicTensorBlock[VALID_VALUE_TYPES_LENGTH];
		_colsdata[that._vt.ordinal()] = that;
	}

	public void reset() {
		reset(_dims);
	}

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
		}
		else {
			for (int i = 0; i < _colsdata.length; i++) {
				if (_colsdata[i] != null) {
					_colsdata[i].reset(toInternalDims(dims, typeIxCounter[i]));
				}
				else if (typeIxCounter[i] != 0) {
					int[] colDims = toInternalDims(_dims, typeIxCounter[i]);
					_colsdata[i] = new BasicTensorBlock(ValueType.values()[i], colDims, false);
					_colsdata[i].allocateBlock();
				}
			}
		}
	}

	public DataTensorBlock allocateBlock() {
		if (_colsdata == null)
			_colsdata = new BasicTensorBlock[VALID_VALUE_TYPES_LENGTH];
		int[] colDataColumnLength = new int[_colsdata.length];
		for (ValueType valueType : _schema)
			colDataColumnLength[valueType.ordinal()]++;
		for (int i = 0; i < _colsdata.length; i++) {
			if (colDataColumnLength[i] != 0) {
				int[] dims = toInternalDims(_dims, colDataColumnLength[i]);
				// TODO sparse
				_colsdata[i] = new BasicTensorBlock(ValueType.values()[i], dims, false);
				_colsdata[i].allocateBlock();
			}
		}
		return this;
	}

	public boolean isAllocated() {
		if (_colsdata == null)
			return false;
		for (BasicTensorBlock block : _colsdata) {
			if (block != null && block.isAllocated())
				return true;
		}
		return false;
	}

	public boolean isEmpty(boolean safe) {
		if (!isAllocated()) {
			return true;
		}
		for (BasicTensorBlock tb : _colsdata) {
			if (tb != null && !tb.isEmpty(safe))
				return false;
		}
		return true;
	}

	public long getNonZeros() {
		if (!isAllocated()) {
			return 0;
		}
		long nnz = 0;
		for (BasicTensorBlock bt : _colsdata) {
			if (bt != null)
				nnz += bt.getNonZeros();
		}
		return nnz;
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

	public ValueType[] getSchema() {
		return _schema;
	}

	public ValueType getColValueType(int col) {
		return _schema[col];
	}

	public Object get(int[] ix) {
		int columns = ix[1];
		int[] internalIx = toInternalIx(ix, _colsToIx[columns]);
		return _colsdata[_schema[columns].ordinal()].get(internalIx);
	}

	public double get(int r, int c) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("DataTensor.get(int,int) dimension mismatch: expected=2 actual=" + getNumDims());
		return _colsdata[_schema[c].ordinal()].get(r, _colsToIx[c]);
	}

	public void set(Object v) {
		for (BasicTensorBlock bt : _colsdata)
			bt.set(v);
	}

	public void set(int[] ix, Object v) {
		int columns = ix[1];
		int[] internalIx = toInternalIx(ix, _colsToIx[columns]);
		_colsdata[_schema[columns].ordinal()].set(internalIx, v);
	}

	public void set(int r, int c, double v) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("DataTensor.set(int,int,double) dimension mismatch: expected=2 actual=" + getNumDims());
		_colsdata[_schema[c].ordinal()].set(r, _colsToIx[c], v);
	}

	public void copy(DataTensorBlock that) {
		_dims = that._dims.clone();
		_schema = that._schema.clone();
		_colsToIx = that._colsToIx.clone();
		_ixToCols = new int[that._ixToCols.length][];
		for (int i = 0; i < _ixToCols.length; i++)
			if (that._ixToCols[i] != null)
				_ixToCols[i] = that._ixToCols[i].clone();
		if (that.isAllocated()) {
			for (int i = 0; i < _colsdata.length; i++) {
				if (that._colsdata[i] != null) {
					_colsdata[i] = new BasicTensorBlock(that._colsdata[i]);
				}
			}
		}
	}
	
	/**
	 * Copy a part of another <code>DataTensorBlock</code>
	 * @param lower lower index of elements to copy (inclusive)
	 * @param upper upper index of elements to copy (exclusive)
	 * @param src source <code>DataTensorBlock</code>
	 */
	public void copy(int[] lower, int[] upper, DataTensorBlock src) {
		int[] subLower = lower.clone();
		if (upper[1] == 0) {
			upper[1] = getDim(1);
			upper[0]--;
		}
		int[] subUpper = upper.clone();
		for (int i = 0; i < VALID_VALUE_TYPES_LENGTH; i++) {
			if (src._colsdata[i] == null)
				continue;
			subLower[1] = lower[1];
			subUpper[1] = lower[1] + src._colsdata[i].getNumColumns();
			_colsdata[i].copy(subLower, subUpper, src._colsdata[i]);
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
