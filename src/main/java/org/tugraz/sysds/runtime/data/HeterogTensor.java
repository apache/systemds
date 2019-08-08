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

import org.apache.commons.lang.ArrayUtils;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

public class HeterogTensor extends Tensor {
	private HomogTensor[] _colsdata = null;
	private ValueType[] _schema = null;

	public HeterogTensor() {
		this(new ValueType[0], DEFAULT_DIMS);
	}

	public HeterogTensor(int ncols, ValueType vt) {
		_dims = new int[]{0, ncols};
		_schema = new ValueType[ncols];
		Arrays.fill(_schema, vt);
	}

	public HeterogTensor(ValueType[] schema) {
		_dims = new int[]{0, schema.length};
		_schema = schema;
	}

	public HeterogTensor(ValueType[] schema, int[] dims) {
		_dims = dims;
		_schema = schema;
		reset();
	}

	public HeterogTensor(ValueType vt, int[] dims) {
		_dims = dims;
		_schema = new ValueType[getDim(1)];
		Arrays.fill(_schema, vt);
		reset();
	}

	public HeterogTensor(ValueType[] schema, int[] dims, String[][] data) {
		_dims = dims;
		_schema = schema;
		allocateBlock();
		for (int i = 0; i < schema.length; i++) {
			fillCol(i, data[i]);
		}
	}

	public HeterogTensor(double val) {
		_dims = new int[]{1, 1};
		_schema = new ValueType[]{ValueType.FP64};
		_colsdata = new HomogTensor[]{new HomogTensor(val)};
	}

	public HeterogTensor(HeterogTensor that) {
		copy(that);
	}

	@Override
	public void reset() {
		reset(_dims);
	}

	@Override
	public void reset(int[] dims) {
		if (dims.length < 2)
			throw new DMLRuntimeException("HeterogTensor.reset(int[]) invalid number of tensor dimensions: " + dims.length);
		if (dims[1] > _dims[1])
			throw new DMLRuntimeException("HeterogTensor.reset(int[]) columns can not be added without a provided schema," +
					" use reset(int[],ValueType[]) instead");
		for (int i = 0; i < dims.length; i++)
			if (dims[i] < 0)
				throw new DMLRuntimeException("HeterogTensor.reset(int[]) invalid  dimension " + i + ": " + dims[i]);
		_dims = dims;
		if (_schema.length < getDim(1))
			_schema = Arrays.copyOfRange(_schema, 0, getDim(1));
		if (_colsdata == null) {
			allocateBlock();
		} else {
			HomogTensor[] newCols = new HomogTensor[getDim(1)];
			if (_colsdata.length > getDim(1))
				_colsdata = Arrays.copyOfRange(_colsdata, 0, getDim(1));
			int[] blockDims = toInternalDims(dims);
			for (HomogTensor colsdata : _colsdata) {
				colsdata.reset(blockDims);
			}
		}
	}

	public void reset(int[] dims, ValueType[] schema) {
		if (dims.length < 2)
			throw new DMLRuntimeException("HeterogTensor.reset(int[],ValueType[]) invalid number of tensor dimensions: " + dims.length);
		if (dims[1] != schema.length)
			throw new DMLRuntimeException("HeterogTensor.reset(int[],ValueType[]) column dimension and schema length does not match");
		for (int i = 0; i < dims.length; i++)
			if (dims[i] < 0)
				throw new DMLRuntimeException("HeterogTensor.reset(int[],ValueType[]) invalid  dimension " + i + ": " + dims[i]);
		_dims = dims;
		_schema = schema;
		if (_colsdata == null) {
			allocateBlock();
		} else {
			HomogTensor[] newCols = new HomogTensor[getDim(1)];
			int[] blockDims = toInternalDims(dims);
			for (int c = 0; c < getDim(1); c++) {
				if (_colsdata[c]._vt != schema[c]) {
					newCols[c] = new HomogTensor(schema[c], blockDims, false);
				}
				else {
					newCols[c] = _colsdata[c];
				}
				newCols[c].reset(blockDims);
			}
			_colsdata = newCols;
		}
	}

	@Override
	public HeterogTensor allocateBlock() {
		if (_colsdata == null)
			_colsdata = new HomogTensor[getDim(1)];
		// All column tensors can share the same dimension array
		// since their dimension should always be equal.
		int[] blockDims = toInternalDims(_dims);
		for (int i = 0; i < _schema.length; i++) {
			// TODO sparse
			_colsdata[i] = new HomogTensor(_schema[i], blockDims, false);
			_colsdata[i].allocateBlock();
		}
		return this;
	}

	@Override
	public boolean isAllocated() {
		if (_colsdata == null)
			return false;
		for (HomogTensor block : _colsdata) {
			if (block.isAllocated())
				return true;
		}
		return false;
	}

	@Override
	public boolean isEmpty(boolean safe) {
		if (!isAllocated()) {
			return true;
		}
		for (HomogTensor tb : _colsdata) {
			if (!tb.isEmpty(safe))
				return false;
		}
		return true;
	}

	public ValueType[] getSchema() {
		return _schema;
	}

	public ValueType getColValueType(int col) {
		return _schema[col];
	}

	@Override
	public double get(int[] ix) {
		int[] internalIx = toInternalIx(ix);
		return _colsdata[ix[1]].get(internalIx);
	}

	@Override
	public double get(int r, int c) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("HeterogTensor.get(int,int) dimension mismatch: expected=2 actual=" + getNumDims());
		return _colsdata[c].get(r, 0);
	}

	@Override
	public long getLong(int[] ix) {
		int[] internalIx = toInternalIx(ix);
		return _colsdata[ix[1]].getLong(internalIx);
	}

	@Override
	public String getString(int[] ix) {
		int[] internalIx = toInternalIx(ix);
		return _colsdata[ix[1]].getString(internalIx);
	}

	@Override
	public void set(int[] ix, double v) {
		int[] internalIx = toInternalIx(ix);
		_colsdata[ix[1]].set(internalIx, v);
	}

	@Override
	public void set(int r, int c, double v) {
		if (getNumDims() != 2)
			throw new DMLRuntimeException("HeterogTensor.set(int,int,double) dimension mismatch: expected=2 actual=" + getNumDims());
		_colsdata[c].set(r, 0, v);
	}

	@Override
	public void set(int[] ix, long v) {
		int[] internalIx = toInternalIx(ix);
		_colsdata[ix[1]].set(internalIx, v);
	}

	@Override
	public void set(int[] ix, String v) {
		int[] internalIx = toInternalIx(ix);
		_colsdata[ix[1]].set(internalIx, v);
	}

	public void copy(HeterogTensor that) {
		_dims = that._dims.clone();
		_schema = that._schema.clone();
		if (that.isAllocated()) {
			allocateBlock();
			if (that.isEmpty(false)) {
				reset();
				return;
			}
			for (int i = 0; i < _colsdata.length; i++) {
				_colsdata[i].copy(that._colsdata[i]);
			}
		}
	}

	public void fillCol(int col, String[] data) {
		int[] ix = new int[getNumDims()];
		for (String datum : data) {
			_colsdata[col].set(ix, datum);
			getNextIndexes(ix);
		}
	}

	public void fillCol(int col, double[] data) {
		int[] ix = new int[getNumDims()];
		for (double datum : data) {
			_colsdata[col].set(ix, datum);
			getNextIndexes(ix);
		}
	}

	public void fillCol(int col, long[] data) {
		int[] ix = new int[getNumDims()];
		for (long datum : data) {
			_colsdata[col].set(ix, datum);
			getNextIndexes(ix);
		}
	}

	public HomogTensor getCol(int col) {
		return _colsdata[col];
	}

	public HeterogTensor appendCol(HomogTensor data) {
		// validate data dimensions
		if (_dims.length != data.getNumDims()) {
			throw new DMLRuntimeException("Append column number of dimensions mismatch: expected=" + _dims.length +
					" actual=" + data.getNumDims());
		}
		if (_dims[0] != data.getDim(0))
			throw new DMLRuntimeException("Append column row dimension mismatch: expected=" + _dims[0] + " actual=" + data.getDim(0));
		for (int i = 2; i < data.getNumDims(); i++) {
			if (_dims[i] != data.getDim(i))
				throw new DMLRuntimeException("Append column " + i + " dimension mismatch: expected=" + _dims[i] +
						" actual=" + data.getDim(i));
		}
		_dims[1]++;
		_schema = (ValueType[]) ArrayUtils.add(_schema, data.getValueType());
		_colsdata = (HomogTensor[]) ArrayUtils.add(_colsdata, data);
		return this;
	}

	@Override
	public long getInMemorySize() {
		// TODO in memory size
		return 0;
	}

	@Override
	public long getExactSerializedSize() {
		//header size (num dims, dims)
		long size = 4 * (1 + _dims.length);
		//colsdata serialized representation
		for (HomogTensor ht : _colsdata)
			size += ht.getExactSerializedSize();
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
		//step 2: write tensors (read schema from those)
		for (int i = 0; i < getDim(1); i++)
			_colsdata[i].write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		//step 1: read dims
		_dims = new int[in.readInt()];
		for (int i = 0; i < _dims.length; i++)
			_dims[i] = in.readInt();
		_schema = new ValueType[getDim(1)];
		_colsdata = new HomogTensor[getDim(1)];
		for (int i = 0; i < getDim(1); i++) {
			_colsdata[i] = new HomogTensor();
			_colsdata[i].readFields(in);
			_schema[i] = _colsdata[i]._vt;
		}
	}

	private static int[] toInternalIx(int[] ix) {
		int[] internalIx = ix.clone();
		internalIx[1] = 0;
		return internalIx;
	}

	private static int[] toInternalDims(int[] dims) {
		int[] internalDims = dims.clone();
		internalDims[1] = 1;
		return internalDims;
	}
}
