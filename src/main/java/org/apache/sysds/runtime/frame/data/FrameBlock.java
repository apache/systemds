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

package org.apache.sysds.runtime.frame.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.lang.ref.SoftReference;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.frame.data.lib.FrameFromMatrixBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibAppend;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.runtime.frame.data.lib.FrameLibDetectSchema;
import org.apache.sysds.runtime.frame.data.lib.FrameLibRemoveEmpty;
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.functionobjects.ValueComparisonFunction;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DMVUtils;
import org.apache.sysds.runtime.util.EMAUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

@SuppressWarnings({"rawtypes", "unchecked"}) // allow generic native arrays
public class FrameBlock implements CacheBlock<FrameBlock>, Externalizable {
	private static final Log LOG = LogFactory.getLog(FrameBlock.class.getName());
	private static final long serialVersionUID = -3993450030207130665L;
	private static final IDSequence CLASS_ID = new IDSequence();
	/** Buffer size variable: 1M elements, size of default matrix block */
	public static final int BUFFER_SIZE = 1 * 1000 * 1000;

	/** If debugging is enabled for the FrameBlocks in stable state */
	public static boolean debug = false;

	/** The schema of the data frame as an ordered list of value types */
	private ValueType[] _schema = null;

	/** The column names of the data frame as an ordered list of strings, allocated on-demand */
	private String[] _colnames = null;

	/** The column metadata */
	private ColumnMetadata[] _colmeta = null;

	/** The data frame data as an ordered list of columns */
	private Array[] _coldata = null;

	/** Locks on the columns not tied to the columns objects. */
	private SoftReference<Object[]> _columnLocks = null;

	/** Materialized number of rows in this FrameBlock */
	private int _nRow = 0;

	/** Cached size in memory to avoid repeated scans of string columns */
	private long _msize = -1;

	public FrameBlock() {
	}

	/**
	 * Copy constructor for frame blocks, which uses a shallow copy for the schema (column types and names) but a deep
	 * copy for meta data and actual column data.
	 *
	 * @param that frame block
	 */
	public FrameBlock(FrameBlock that) {
		this(that.getSchema(), that.getColumnNames(false));
		copy(that);
		setColumnMetadata(that.getColumnMetadata());
	}

	public FrameBlock(int ncols, ValueType vt) {
		this(UtilFunctions.nCopies(ncols, vt), null, null);
	}

	public FrameBlock(ValueType[] schema) {
		this(schema, null, null);
	}

	public FrameBlock(ValueType[] schema, int rlen) {
		this(schema, null, null);
		_nRow = rlen;
	}

	public FrameBlock(ValueType[] schema, String[] names) {
		this(schema, names, null);
	}

	public FrameBlock(ValueType[] schema, String[] names, int rlen) {
		this(schema, names, null);
		_nRow = rlen;
	}

	public FrameBlock(ValueType[] schema, String[][] data) {
		// default column names not materialized
		this(schema, null, data);
	}

	/**
	 * FrameBlock constructor with constant
	 * 
	 * @param schema   The schema to allocate (also specifying number of columns)
	 * @param constant The constant to allocate in all cells
	 * @param nRow     the number of rows
	 */
	public FrameBlock(ValueType[] schema, String constant, int nRow) {
		this();
		// allocate the values.
		_nRow = nRow;
		for(int i = 0; i < schema.length; i++)
			appendColumn(ArrayFactory.allocate(schema[i], nRow, constant));
	}

	/**
	 * allocate a FrameBlock with the given data arrays.
	 * 
	 * The data is in row major, making the first dimension number of rows. second number of columns.
	 * 
	 * @param schema the schema to allocate
	 * @param names  The names of the column
	 * @param data   The data.
	 */
	public FrameBlock(ValueType[] schema, String[] names, String[][] data) {
		_schema = schema;
		if(names != null) {
			_colnames = names;
			if(schema.length != names.length)
				throw new DMLRuntimeException("Invalid FrameBlock construction, invalid schema and names combination");
		}
		ensureAllocateMeta();
		if(data != null) {
			for(int i = 0; i < data.length; i++)
				appendRow(data[i]);
		}
	}

	public FrameBlock(ValueType[] schema, String[] colNames, ColumnMetadata[] meta, Array<?>[] data) {
		_schema = schema;
		_colnames = colNames;
		_colmeta = meta;
		_coldata = data;
		_nRow = data[0].size();
	}

	/**
	 * Create a FrameBlock containing columns of the specified arrays
	 * 
	 * @param data The column data contained
	 */
	public FrameBlock(Array<?>[] data) {
		_schema = new ValueType[data.length];
		for(int i = 0; i < data.length; i++)
			_schema[i] = data[i].getValueType();

		_colnames = null;
		ensureAllocateMeta();
		_coldata = data;
		_nRow = data[0].size();

		if(debug) {
			for(int i = 0; i < data.length; i++) {
				if(data[i].size() != getNumRows())
					throw new DMLRuntimeException("Invalid Frame allocation with different size arrays "
						+ data[i].size() + " vs " + getNumRows());
			}
		}
	}

	/**
	 * Create a FrameBlock containing columns of the specified arrays and names
	 * 
	 * @param data     The column data contained
	 * @param colnames The column names of the contained columns
	 */
	public FrameBlock(Array<?>[] data, String[] colnames) {
		_schema = new ValueType[data.length];
		for(int i = 0; i < data.length; i++)
			_schema[i] = data[i].getValueType();

		_colnames = colnames;
		ensureAllocateMeta();
		_coldata = data;
		_nRow = data[0].size();

		if(debug) {
			for(int i = 0; i < data.length; i++) {
				if(data[i].size() != getNumRows())
					throw new DMLRuntimeException("Invalid Frame allocation with different size arrays "
						+ data[i].size() + " vs " + getNumRows());
			}
		}
	}

	/**
	 * Get the number of rows of the frame block.
	 *
	 * @return number of rows
	 */
	@Override
	public int getNumRows() {
		return _nRow;
	}

	@Override
	public double getDouble(int r, int c) {
		return _coldata[c].getAsDouble(r);
	}

	@Override
	public double getDoubleNaN(int r, int c) {
		return _coldata[c].getAsNaNDouble(r);
	}

	@Override
	public String getString(int r, int c) {
		Object o = get(r, c);
		String s = (o == null) ? null : o.toString();
		if(s != null && s.isEmpty())
			return null;
		return s;
	}

	/**
	 * Get the number of columns of the frame block, that is the number of columns defined in the schema.
	 *
	 * @return number of columns
	 */
	@Override
	public int getNumColumns() {
		return (_schema != null) ? _schema.length : 0;
	}

	@Override
	public DataCharacteristics getDataCharacteristics() {
		return new MatrixCharacteristics(getNumRows(), getNumColumns(), -1);
	}

	/**
	 * Returns the schema of the frame block.
	 *
	 * @return schema as array of ValueTypes
	 */
	public ValueType[] getSchema() {
		return _schema;
	}

	/**
	 * Sets the schema of the frame block.
	 *
	 * @param schema schema as array of ValueTypes
	 */
	public void setSchema(ValueType[] schema) {
		_schema = schema;
	}

	/**
	 * Returns the column names of the frame block. This method allocates default column names if required.
	 *
	 * @return column names
	 */
	public String[] getColumnNames() {
		return getColumnNames(true);
	}

	public FrameBlock getColumnNamesAsFrame() {
		FrameBlock fb = new FrameBlock(getNumColumns(), ValueType.STRING);
		fb.appendRow(getColumnNames());
		return fb;
	}

	/**
	 * Returns the column names of the frame block. This method allocates default column names if required.
	 *
	 * @param alloc if true, create column names
	 * @return array of column names
	 */
	public String[] getColumnNames(boolean alloc) {
		if(_colnames == null && alloc)
			_colnames = createColNames(getNumColumns());
		return _colnames;
	}

	/**
	 * Returns the column name for the requested column. This method allocates default column names if required.
	 *
	 * @param c column index
	 * @return column name
	 */
	public String getColumnName(int c) {
		if(_colnames == null)
			_colnames = createColNames(getNumColumns());
		return _colnames[c];
	}

	public void setColumnNames(String[] colnames) {
		_colnames = colnames;
	}

	public void setColumnName(int index, String name) {
		if(_colnames == null)
			_colnames = createColNames(getNumColumns());
		_colnames[index] = name;
	}

	public ColumnMetadata[] getColumnMetadata() {
		return _colmeta;
	}

	public ColumnMetadata getColumnMetadata(int c) {
		return _colmeta[c];
	}

	public Array<?>[] getColumns() {
		return _coldata;
	}

	public boolean isColumnMetadataDefault() {
		boolean ret = true;
		for(int j = 0; j < getNumColumns() && ret; j++)
			ret &= isColumnMetadataDefault(j);
		return ret;
	}

	public boolean isColumnMetadataDefault(int c) {
		return _colmeta[c].isDefault();
	}

	public void setColumnMetadata(ColumnMetadata[] colmeta) {
		System.arraycopy(colmeta, 0, _colmeta, 0, _colmeta.length);
	}

	public void setColumnMetadata(int c, ColumnMetadata colmeta) {
		_colmeta[c] = colmeta;
	}

	/**
	 * Creates a mapping from column names to column IDs, i.e., 1-based column indexes
	 *
	 * @return map of column name keys and id values
	 */
	public Map<String, Integer> getColumnNameIDMap() {
		Map<String, Integer> ret = new HashMap<>();
		for(int j = 0; j < getNumColumns(); j++)
			ret.put(getColumnName(j), j + 1);
		return ret;
	}

	/**
	 * Allocate column data structures if necessary, i.e., if schema specified but not all column data structures
	 * created yet.
	 *
	 * @param numRows number of rows
	 */
	public void ensureAllocatedColumns(int numRows) {
		_msize = -1;
		final int nRow = getNumRows();
		// allocate column meta data if necessary
		ensureAllocateMeta();
		// early abort if already allocated
		if(_coldata != null && _schema.length == _coldata.length) {
			// handle special case that to few rows allocated
			if(nRow < numRows) {
				String[] tmp = new String[getNumColumns()];
				int len = numRows - nRow;
				// TODO: Add append N function.
				for(int i = 0; i < len; i++)
					appendRow(tmp);
			}
			return;
		}
		else {

			// allocate columns if necessary
			_coldata = new Array[_schema.length];
			if(numRows > 0)
				for(int j = 0; j < _schema.length; j++)
					_coldata[j] = ArrayFactory.allocate(_schema[j], numRows);
			_nRow = numRows;
		}
	}

	private void ensureAllocateMeta() {
		if(_colmeta == null || _schema.length != _colmeta.length) {
			_colmeta = new ColumnMetadata[_schema.length];
			for(int j = 0; j < _schema.length; j++)
				_colmeta[j] = new ColumnMetadata();
		}
	}

	/**
	 * Checks for matching column sizes in case of existing columns.
	 * 
	 * If the check parses the number of rows is reassigned to the given newLen
	 *
	 * @param newLen number of rows to compare with existing number of rows
	 */
	public void ensureColumnCompatibility(int newLen) {
		final int nRow = getNumRows();
		if(_coldata != null && _coldata.length > 0 && ((nRow == 0) || nRow != newLen)) {
			throw new RuntimeException("Mismatch in number of rows: " + newLen + " (expected: " + nRow + ")");
		}
		_nRow = newLen;
	}

	public static String[] createColNames(int size) {
		return createColNames(0, size);
	}

	public static String[] createColNames(int off, int size) {
		String[] ret = new String[size];
		for(int i = off + 1; i <= off + size; i++)
			ret[i - off - 1] = createColName(i);
		return ret;
	}

	public static String createColName(int i) {
		return "C" + i;
	}

	public boolean isColNamesDefault() {
		boolean ret = (_colnames != null);
		for(int j = 0; j < getNumColumns() && ret; j++)
			ret &= isColNameDefault(j);
		return ret;
	}

	public boolean isColNameDefault(int i) {
		return _colnames == null || _colnames[i].equals("C" + (i + 1));
	}

	public void recomputeColumnCardinality() {
		for(int j = 0; j < getNumColumns(); j++) {
			int card = 0;
			for(int i = 0; i < getNumRows(); i++)
				card += (get(i, j) != null) ? 1 : 0;
			_colmeta[j].setNumDistinct(card);
		}
	}

	///////
	// basic get and set functionality

	/**
	 * Gets a boxed object of the value in position (r,c).
	 *
	 * @param r row index, 0-based
	 * @param c column index, 0-based
	 * @return object of the value at specified position
	 */
	public Object get(int r, int c) {
		return _coldata[c].get(r);
	}

	/**
	 * Sets the value in position (r,c), where the input is assumed to be a boxed object consistent with the schema
	 * definition.
	 *
	 * @param r   row index
	 * @param c   column index
	 * @param val value to set at specified position
	 */
	public void set(int r, int c, Object val) {
		_coldata[c].set(r, UtilFunctions.objectToObject(_schema[c], val));
	}

	/**
	 * Sets the value in position (r,c), to the input string value, and at the individual arrays, convert to correct
	 * type.
	 * 
	 * @param r   row index
	 * @param c   column index
	 * @param val value to set at specified position
	 */
	public void set(int r, int c, String val) {
		_coldata[c].set(r, val);
	}

	public void reset(int nrow, boolean clearMeta) {
		if(clearMeta) {
			_schema = null;
			_colnames = null;
			if(_colmeta != null) {
				for(int i = 0; i < _colmeta.length; i++)
					if(!isColumnMetadataDefault(i))
						_colmeta[i] = new ColumnMetadata();
			}
		}
		if(_coldata != null) {
			for(int i = 0; i < _coldata.length; i++)
				_coldata[i].reset(nrow);
		}
		_nRow = nrow;
		_msize = -1;
	}

	public void reset() {
		reset(0, true);
	}

	/**
	 * Sets row at position r to the input array of objects, corresponding to the schema.
	 * @param r	  row index
	 * @param row array of objects
	 */
	public void setRow(int r, Object[] row) {
		for (int i = 0; i < row.length; i++) {
			set(r, i, row[i]);
		}
	}

	/**
	 * Append a row to the end of the data frame, where all row fields are boxed objects according to the schema.
	 *
	 * Append row should be avoided if possible.
	 * 
	 * @param row array of objects
	 */
	public void appendRow(Object[] row) {
		if(row.length != _schema.length)
			throw new DMLRuntimeException("Invalid number of values in rowAppend");
		if(_nRow == 0) {
			ensureAllocateMeta();
			_coldata = new Array[_schema.length];
			for(int j = 0; j < _schema.length; j++) {
				_coldata[j] = ArrayFactory.allocate(_schema[j], 1);
				_coldata[j].set(0, row[j]);
			}
		}
		else {
			for(int j = 0; j < row.length; j++)
				_coldata[j].append(row[j]);
		}
		_nRow++;
		_msize = -1;
	}

	/**
	 * Append a row to the end of the data frame, where all row fields are string encoded.
	 *
	 * Append row should be avoided if possible
	 * 
	 * @param row array of strings
	 */
	public void appendRow(String[] row) {
		if(row.length != _schema.length)
			throw new DMLRuntimeException("Invalid number of values in rowAppend");
		else if(_nRow == 0) {
			ensureAllocateMeta();
			_coldata = new Array[_schema.length];
			for(int j = 0; j < _schema.length; j++) {
				_coldata[j] = ArrayFactory.allocate(_schema[j], 1);
				_coldata[j].set(0, row[j]);
			}
		}
		else {
			for(int j = 0; j < row.length; j++)
				_coldata[j].append(row[j]);
		}
		_nRow++;
		_msize = -1;
	}

	/**
	 * Append a column of value type STRING as the last column of the data frame. The given array is wrapped but not
	 * copied and hence might be updated in the future.
	 *
	 * @param col array of strings
	 */
	public void appendColumn(String[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.STRING);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append a column of value type BOOLEAN as the last column of the data frame. The given array is wrapped but not
	 * copied and hence might be updated in the future.
	 *
	 * @param col array of booleans
	 */
	public void appendColumn(boolean[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.BOOLEAN);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append a column of value type INT as the last column of the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of longs
	 */
	public void appendColumn(int[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.INT32);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append a column of value type LONG as the last column of the data frame. The given array is wrapped but not
	 * copied and hence might be updated in the future.
	 *
	 * @param col array of longs
	 */
	public void appendColumn(long[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.INT64);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append a column of value type float as the last column of the data frame. The given array is wrapped but not
	 * copied and hence might be updated in the future.
	 *
	 * @param col array of doubles
	 */
	public void appendColumn(float[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.FP32);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append a column of value type DOUBLE as the last column of the data frame. The given array is wrapped but not
	 * copied and hence might be updated in the future.
	 *
	 * @param col array of doubles
	 */
	public void appendColumn(double[] col) {
		ensureColumnCompatibility(col.length);
		appendColumnMetaData(ValueType.FP64);
		_coldata = FrameUtil.add(_coldata, ArrayFactory.create(col));
	}

	/**
	 * Append the metadata associated with adding a column.
	 * 
	 * @param vt The Value type
	 */
	private void appendColumnMetaData(ValueType vt) {
		if(_colnames != null)
			_colnames = ArrayUtils.add(getColumnNames(), createColName(_colnames.length + 1));
		_schema = ArrayUtils.add(_schema, vt);
		_colmeta = ArrayUtils.add(getColumnMetadata(), new ColumnMetadata());
		_msize = -1;
	}

	/**
	 * Append a set of column of value type DOUBLE at the end of the frame in order to avoid repeated allocation with
	 * appendColumns. The given array is wrapped but not copied and hence might be updated in the future.
	 *
	 * @param cols 2d array of doubles
	 */
	public void appendColumns(double[][] cols) {
		int ncol = cols.length;
		boolean empty = (_schema == null);
		ValueType[] tmpSchema = UtilFunctions.nCopies(ncol, ValueType.FP64);
		Array[] tmpData = new Array[ncol];
		for(int j = 0; j < ncol; j++)
			tmpData[j] = ArrayFactory.create(cols[j]);
		_colnames = empty ? null : ArrayUtils.addAll(getColumnNames(), createColNames(getNumColumns(), ncol)); // before
																												// schema
																												// modification
		_schema = empty ? tmpSchema : ArrayUtils.addAll(_schema, tmpSchema);
		_coldata = empty ? tmpData : ArrayUtils.addAll(_coldata, tmpData);
		_nRow = cols[0].length;
		_msize = -1;
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType[] schema, int k) {
		return FrameFromMatrixBlock.convertToFrameBlock(mb, schema, k);
	}

	/**
	 * Add a column of already allocated Array type.
	 * 
	 * @param col column to add.
	 */
	public void appendColumn(Array col) {
		ensureColumnCompatibility(col.size());
		appendColumnMetaData(col.getValueType());
		_coldata = FrameUtil.add(_coldata, col);
	}

	public Object getColumnData(int c) {
		return _coldata[c].get();
	}

	public ValueType getColumnType(int c) {
		return _schema[c];
	}

	public Array<?> getColumn(int c) {
		return _coldata != null ? _coldata[c] : null;
	}

	public void setColumn(int c, Array<?> column) {
		if(_coldata == null) {
			_coldata = new Array[getNumColumns()];
			if(column != null)
				_nRow = column.size();
		}
		else if(column != null && column.size() != _nRow)
			throw new DMLRuntimeException("Invalid number of rows in set column");
		_coldata[c] = column;
		_msize = -1;
	}

	/**
	 * Appends a chunk of data to the end of a specified column.
	 * 
	 * @param c     column index
	 * @param chunk chunk of data to append
	 */
	public void appendColumnChunk(int c, Array<?> chunk) {
		if (_coldata == null) {
			_coldata = new Array[getNumColumns()];
		}

		if (_coldata[c] == null) {
			_coldata[c] = chunk;
			_nRow = chunk.size();
		} else {
			_coldata[c] = ArrayFactory.append(_coldata[c], chunk);
			_nRow += chunk.size();
		}

		_msize = -1;
	}

	/**
	 * Sets a chunk of data to a specified column, starting at the specified offset.
	 * 
	 * @param c		  column index
	 * @param chunk   chunk of data to set
	 * @param offset  offset position where it should set the chunk
	 * @param colSize size of columns, in case columns aren't initialized yet
	 */
	public void setColumnChunk(int c, Array<?> chunk, int offset, int colSize) {
		if (_coldata == null) {
			_coldata = new Array[getNumColumns()];
			_nRow = colSize;
		}

		if (_coldata[c] == null) {
			_coldata[c] = ArrayFactory.allocate(chunk.getValueType(), _nRow);
		}

		if (_coldata[c].getValueType() != chunk.getValueType()) {
			throw new DMLRuntimeException("ValueType mismatch in setColumnChunk: expected " +
					_coldata[c].getValueType() + " but got " + chunk.getValueType());
		}

		ArrayFactory.set(_coldata[c], chunk, offset, offset + chunk.size() - 1, _nRow);
	}


	@Override
	public void write(DataOutput out) throws IOException {
		final boolean isDefaultMeta = isColNamesDefault() && isColumnMetadataDefault();
		// write header (rows, cols, default)
		final int nRow = getNumRows();
		out.writeInt(nRow);
		out.writeInt(getNumColumns());
		out.writeBoolean(isDefaultMeta);
		// write columns (value type, data)
		for(int j = 0; j < getNumColumns(); j++) {
			final byte type = getTypeForIO(j);
			out.writeByte(type);
			if(!isDefaultMeta) {
				out.writeUTF(getColumnName(j));
				_colmeta[j].write(out);
			}
			if(type > 0 && nRow > 0) // if allocated write column data
				_coldata[j].write(out);
		}
	}

	private byte getTypeForIO(int col) {
		// ! +1 to allow reflecting around zero if not allocated
		byte type = (byte) (_schema[col].ordinal() + 1);
		if(_coldata == null || _coldata[col] == null)
			type *= -1; // negative to indicate not allocated
		return type;
	}

	private ValueType interpretByteAsType(byte type) {
		return ValueType.values()[Math.abs(type) - 1];
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		// read head (rows, cols)
		_nRow = in.readInt();
		final int numCols = in.readInt();
		final boolean isDefaultMeta = in.readBoolean();
		// allocate schema/meta data arrays
		_schema = (_schema != null && _schema.length == numCols) ? _schema : new ValueType[numCols];
		_colnames = (_colnames != null && _colnames.length == numCols) ? _colnames : // if already allocated reuse
			isDefaultMeta ? null : new String[numCols]; // if meta is default allocate on demand
		_colmeta = (_colmeta != null && _colmeta.length == numCols) ? _colmeta : new ColumnMetadata[numCols];
		_coldata = (_coldata != null && _coldata.length == numCols) ? _coldata : new Array[numCols];
		if(_nRow == 0)
			_coldata = null;
		// read columns (value type, meta, data)
		for(int j = 0; j < numCols; j++) {
			byte type = in.readByte();
			_schema[j] = interpretByteAsType(type);
			if(!isDefaultMeta) { // If not default meta read in meta
				_colnames[j] = in.readUTF();
				_colmeta[j] = ColumnMetadata.read(in);
			}
			else
				_colmeta[j] = new ColumnMetadata(); // must be allocated.

			if(type >= 0 && _nRow > 0) // if in allocated column data then read it
				_coldata[j] = ArrayFactory.read(in, _nRow);
		}
		_msize = -1;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		readFields(in);
	}

	@Override
	public long getInMemorySize() {
		// reuse previously computed size
		if(_msize > 0)
			return _msize;

		// frame block header
		double size = 16 + 4; // object, msize

		final int clen = getNumColumns();

		// schema array (overhead and int entries)
		size += MemoryEstimates.byteArrayCost(clen);

		// col name array (overhead and string entries)
		size += _colnames == null ? 8 : MemoryEstimates.stringArrayCost(_colnames);

		// meta data array (overhead and entries)
		size += MemoryEstimates.objectArrayCost(clen);
		if( _colmeta != null )
			for(ColumnMetadata mtd : _colmeta)
				size += mtd == null ? 8 : mtd.getInMemorySize();

		// data array
		size += MemoryEstimates.objectArrayCost(clen);
		size += arraysSizeInMemory();

		return _msize = (long) size;
	}

	private double arraysSizeInMemory() {
		final int clen = getNumColumns();
		final int rlen = getNumRows();
		double size = 0;
		if(_coldata == null) // not allocated estimate if allocated
			for(int j = 0; j < clen; j++)
				size += ArrayFactory.getInMemorySize(_schema[j], rlen, true);
		else {// allocated
			if((rlen > 1000 || clen > 10 )&& ConfigurationManager.isParallelIOEnabled()) {
				final ExecutorService pool = CommonThreadPool.get();
				try {
					List<Future<Long>> f = new ArrayList<>(clen);
					for(int i = 0; i < clen; i++) {
						final int j = i;
						f.add(pool.submit(() -> _coldata[j].getInMemorySize()));
					}

					for(Future<Long> e : f) {
						size += e.get();
					}
				}
				catch(InterruptedException | ExecutionException e) {
					LOG.error(e);
					size = 0;
					for(Array<?> aa : _coldata)
						size += aa.getInMemorySize();
				}
				finally {
					pool.shutdown();
				}
			}
			else {
				for(Array<?> aa : _coldata)
					size += aa.getInMemorySize();
			}
		}
		return size;
	}

	@Override
	public long getExactSerializedSize() {
		// header: 2 x int, boolean
		long size = 4 + 4 + 1;

		size += 1 * getNumColumns(); // column schema
		// column sizes
		final boolean isDefaultMeta = isColNamesDefault() && isColumnMetadataDefault();
		for(int j = 0; j < getNumColumns(); j++) {
			final byte type = getTypeForIO(j);
			if(!isDefaultMeta) {
				size += IOUtilFunctions.getUTFSize(getColumnName(j));
				size += _colmeta[j].getExactSerializedSize();
			}
			if(type > 0)
				size += _coldata[j].getExactSerializedSize();
		}
		return size;
	}

	@Override
	public boolean isShallowSerialize() {
		return isShallowSerialize(false);
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		// shallow serialize if non-string schema because a frame block
		// is always dense but strings have large array overhead per cell
		if( _schema != null )
			for(int j = 0; j < _schema.length; j++)
				if(!_coldata[j].isShallowSerialize())
					return false;
		return true;
	}

	@Override
	public void toShallowSerializeBlock() {
		// do nothing (not applicable).
	}

	@Override
	public void compactEmptyBlock() {
		// do nothing
	}

	/**
	 * This method performs the value comparison on two frames if the values in both frames are equal, not equal, less
	 * than, greater than, less than/greater than and equal to the output frame will store boolean value for each each
	 * comparison
	 *
	 * @param bop  binary operator
	 * @param that frame block of rhs of m * n dimensions
	 * @param out  output frame block
	 * @return a boolean frameBlock
	 */
	public FrameBlock binaryOperations(BinaryOperator bop, FrameBlock that, FrameBlock out) {
		if(getNumColumns() != that.getNumColumns() && getNumRows() != that.getNumColumns())
			throw new DMLRuntimeException("Frame dimension mismatch " + getNumRows() + " * " + getNumColumns() + " != "
				+ that.getNumRows() + " * " + that.getNumColumns());
		String[][] outputData = new String[getNumRows()][getNumColumns()];
		// compare output value, incl implicit type promotion if necessary
		if(bop.fn instanceof ValueComparisonFunction) {
			ValueComparisonFunction vcomp = (ValueComparisonFunction) bop.fn;
			out = executeValueComparisons(this, that, vcomp, outputData);
		}
		else
			throw new DMLRuntimeException("Unsupported binary operation on frames (only comparisons supported)");

		return out;
	}

	private FrameBlock executeValueComparisons(FrameBlock frameBlock, FrameBlock that, ValueComparisonFunction vcomp,
		String[][] outputData) {
		for(int i = 0; i < getNumColumns(); i++) {
			if(getSchema()[i] == ValueType.STRING || that.getSchema()[i] == ValueType.STRING) {
				for(int j = 0; j < getNumRows(); j++) {
					if(checkAndSetEmpty(frameBlock, that, outputData, j, i))
						continue;
					String v1 = UtilFunctions.objectToString(get(j, i));
					String v2 = UtilFunctions.objectToString(that.get(j, i));
					outputData[j][i] = String.valueOf(vcomp.compare(v1, v2));
				}
			}
			else if(getSchema()[i] == ValueType.FP64 || that.getSchema()[i] == ValueType.FP64 ||
				getSchema()[i] == ValueType.FP32 || that.getSchema()[i] == ValueType.FP32) {
				for(int j = 0; j < getNumRows(); j++) {
					if(checkAndSetEmpty(frameBlock, that, outputData, j, i))
						continue;
					ScalarObject so1 = new DoubleObject(Double.parseDouble(get(j, i).toString()));
					ScalarObject so2 = new DoubleObject(Double.parseDouble(that.get(j, i).toString()));
					outputData[j][i] = String.valueOf(vcomp.compare(so1.getDoubleValue(), so2.getDoubleValue()));
				}
			}
			else if(getSchema()[i] == ValueType.INT64 || that.getSchema()[i] == ValueType.INT64 ||
				getSchema()[i] == ValueType.INT32 || that.getSchema()[i] == ValueType.INT32) {
				for(int j = 0; j < this.getNumRows(); j++) {
					if(checkAndSetEmpty(frameBlock, that, outputData, j, i))
						continue;
					ScalarObject so1 = new IntObject(Integer.parseInt(get(j, i).toString()));
					ScalarObject so2 = new IntObject(Integer.parseInt(that.get(j, i).toString()));
					outputData[j][i] = String.valueOf(vcomp.compare(so1.getLongValue(), so2.getLongValue()));
				}
			}
			else {
				for(int j = 0; j < getNumRows(); j++) {
					if(checkAndSetEmpty(frameBlock, that, outputData, j, i))
						continue;
					ScalarObject so1 = new BooleanObject(Boolean.parseBoolean(get(j, i).toString()));
					ScalarObject so2 = new BooleanObject(Boolean.parseBoolean(that.get(j, i).toString()));
					outputData[j][i] = String.valueOf(vcomp.compare(so1.getBooleanValue(), so2.getBooleanValue()));
				}
			}
		}
		return new FrameBlock(UtilFunctions.nCopies(frameBlock.getNumColumns(), ValueType.BOOLEAN), outputData);
	}

	private static boolean checkAndSetEmpty(FrameBlock fb1, FrameBlock fb2, String[][] out, int r, int c) {
		if(fb1.get(r, c) == null || fb2.get(r, c) == null) {
			out[r][c] = (fb1.get(r, c) == null && fb2.get(r, c) == null) ? "true" : "false";
			return true;
		}
		return false;
	}

	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, IndexRange ixrange, FrameBlock ret) {
		return leftIndexingOperations(rhsFrame, (int) ixrange.rowStart, (int) ixrange.rowEnd, (int) ixrange.colStart,
			(int) ixrange.colEnd, ret);
	}

	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, int rl, int ru, int cl, int cu, FrameBlock ret) {
		// check the validity of bounds
		if(rl < 0 || rl >= getNumRows() || ru < rl || ru >= getNumRows() || cl < 0 || cu >= getNumColumns() ||
			cu < cl || cu >= getNumColumns()) {
			throw new DMLRuntimeException(
				"Invalid values for frame indexing: [" + (rl + 1) + ":" + (ru + 1) + "," + (cl + 1) + ":" + (cu + 1)
					+ "] " + "must be within frame dimensions [" + getNumRows() + "," + getNumColumns() + "].");
		}

		if((ru - rl + 1) < rhsFrame.getNumRows() || (cu - cl + 1) < rhsFrame.getNumColumns()) {
			throw new DMLRuntimeException(
				"Invalid values for frame indexing: " + "dimensions of the source frame [" + rhsFrame.getNumRows() + "x"
					+ rhsFrame.getNumColumns() + "] " + "do not match the shape of the frame specified by indices ["
					+ (rl + 1) + ":" + (ru + 1) + ", " + (cl + 1) + ":" + (cu + 1) + "].");
		}

		// allocate output frame (incl deep copy schema)
		if(ret == null)
			ret = new FrameBlock();

		ret._schema = _schema.clone();
		ret._colnames = (_colnames != null) ? _colnames.clone() : null;
		ret._colmeta = _colmeta.clone();
		ret._coldata = new Array[getNumColumns()];
		ret._nRow = _nRow;

		// copy data to output and partial overwrite w/ rhs
		for(int j = 0; j < getNumColumns(); j++) {
			Array tmp = _coldata[j].clone();
			if(j >= cl && j <= cu) {
				// fast-path for homogeneous column schemas
				if(_schema[j] == rhsFrame._schema[j - cl])
					tmp.set(rl, ru, rhsFrame._coldata[j - cl]);
				// general-path for heterogeneous column schemas
				else {
					for(int i = rl; i <= ru; i++)
						tmp.set(i, UtilFunctions.objectToObject(_schema[j], rhsFrame._coldata[j - cl].get(i - rl)));
				}
			}
			ret._coldata[j] = tmp;
		}

		return ret;
	}

	@Override
	public final FrameBlock slice(IndexRange ixrange, FrameBlock ret) {
		return slice((int) ixrange.rowStart, (int) ixrange.rowEnd, (int) ixrange.colStart, (int) ixrange.colEnd, ret);
	}

	@Override
	public final FrameBlock slice(int rl, int ru) {
		return slice(rl, ru, 0, getNumColumns() - 1, false, null);
	}

	@Override
	public final FrameBlock slice(int rl, int ru, boolean deep) {
		return slice(rl, ru, 0, getNumColumns() - 1, deep, null);
	}

	@Override
	public final FrameBlock slice(int rl, int ru, int cl, int cu) {
		return slice(rl, ru, cl, cu, false, null);
	}

	@Override
	public final FrameBlock slice(int rl, int ru, int cl, int cu, FrameBlock ret) {
		return slice(rl, ru, cl, cu, false, ret);
	}

	@Override
	public final FrameBlock slice(int rl, int ru, int cl, int cu, boolean deep) {
		return slice(rl, ru, cl, cu, deep, null);
	}

	@Override
	public FrameBlock slice(int rl, int ru, int cl, int cu, boolean deep, FrameBlock ret) {
		validateSliceArgument(rl, ru, cl, cu);
		// allocate output frame
		if(ret == null)
			ret = new FrameBlock();

		// copy output schema and colnames
		int numCols = cu - cl + 1;
		boolean isDefNames = isColNamesDefault();
		ret._nRow = ru - rl + 1;
		ret._schema = new ValueType[numCols];
		ret._colnames = !isDefNames ? new String[numCols] : null;
		ret._colmeta = new ColumnMetadata[numCols];

		// names
		for(int j = cl; j <= cu; j++) {
			ret._schema[j - cl] = _schema[j];
			ret._colmeta[j - cl] = _colmeta[j];
			if(!isDefNames)
				ret._colnames[j - cl] = getColumnName(j);
		}

		if(ret._coldata == null)
			ret._coldata = new Array[numCols];

		// fast-path: shallow copy column indexing
		if(ret.getNumRows() == getNumRows() && !deep) {
			// this shallow copy does not only avoid an array copy, but
			// also allows for bi-directional reuses of recodemaps
			for(int j = cl; j <= cu; j++)
				ret._coldata[j - cl] = _coldata[j];
		}
		// copy output data
		else {
			for(int j = cl; j <= cu; j++) {
				if(ret._coldata[j - cl] == null)
					ret._coldata[j - cl] = _coldata[j].slice(rl, ru + 1);
				else
					ret._coldata[j - cl].set(0, ru - rl, _coldata[j], rl);
			}
		}

		return ret;
	}

	protected void validateSliceArgument(int rl, int ru, int cl, int cu) {
		if(rl < 0 || rl >= getNumRows() || ru < rl || ru >= getNumRows() || cl < 0 || cu >= getNumColumns() ||
			cu < cl || cu >= getNumColumns()) {
			throw new DMLRuntimeException(
				"Invalid values for frame indexing: [" + (rl + 1) + ":" + (ru + 1) + "," + (cl + 1) + ":" + (cu + 1)
					+ "] " + "must be within frame dimensions [" + getNumRows() + "," + getNumColumns() + "]");
		}
	}

	public void slice(ArrayList<Pair<Long, FrameBlock>> outList, IndexRange range, int rowCut) {
		if(getNumRows() > 0) {
			if(outList.size() > 1)
				throw new NotImplementedException("Not implemented slice of more than 1 block out");
			int r = (int) range.rowStart;
			final FrameBlock out = outList.get(0).getValue();
			if(range.rowStart < rowCut)
				slice(r, (int) Math.min(rowCut, range.rowEnd + 1), (int) range.colStart, (int) range.colEnd, out);

			if(range.rowEnd >= rowCut)
				slice(r, (int) range.rowEnd, (int) range.colStart, (int) range.colEnd, out);

		}
	}

	/**
	 * Appends the given argument FrameBlock 'that' to this FrameBlock by creating a deep copy to prevent side effects.
	 * For cbind, the frames are appended column-wise (same number of rows), while for rbind the frames are appended
	 * row-wise (same number of columns).
	 *
	 * @param that  frame block to append to current frame block
	 * @param cbind if true, column append
	 * @return frame block
	 */
	public FrameBlock append(FrameBlock that, boolean cbind) {
		return FrameLibAppend.append(this, that, cbind);
	}

	public FrameBlock copy() {
		FrameBlock ret = new FrameBlock();
		ret.copy(this);
		return ret;
	}

	public void copy(FrameBlock src) {
		int nCol = src.getNumColumns();
		_nRow = src.getNumRows();
		_schema = Arrays.copyOf(src._schema, nCol);
		if(src._colnames != null)
			_colnames = Arrays.copyOf(src._colnames, nCol);
		if(!src.isColumnMetadataDefault())
			_colmeta = Arrays.copyOf(src._colmeta, nCol);
		if(src._coldata != null) {
			_coldata = new Array<?>[nCol];
			for(int i = 0; i < nCol; i++)
				_coldata[i] = src._coldata[i].clone();
		}
		_msize = -1;
	}

	public FrameBlock copyShallow(){
		FrameBlock ret = new FrameBlock();
		ret._nRow = _nRow;
		ret._msize = _msize; 
		final int nCol = getNumColumns();
		if(_coldata != null)
			ret._coldata = Arrays.copyOf(_coldata, nCol);
		if(_colnames != null)
			ret._colnames = Arrays.copyOf(_colnames, nCol);
		if(_colmeta != null)
			ret._colmeta = Arrays.copyOf(_colmeta, nCol);
		if(_schema != null)
			ret._schema = Arrays.copyOf(_schema, nCol);
		return ret;
	}

	/**
	 * Copy src matrix into the index range of the existing current matrix.
	 *
	 * This is used to copy smaller blocks into a larger block, for instance in binary reading.
	 * 
	 * @param rl  row start
	 * @param ru  row end inclusive
	 * @param cl  col start
	 * @param cu  col end inclusive
	 * @param src source FrameBlock typically a smaller block.
	 */
	public void copy(int rl, int ru, int cl, int cu, FrameBlock src) {
		// If full copy, fall back to default copy
		if(rl == 0 && cl == 0 && ru + 1 == this.getNumRows() && cu + 1 == this.getNumColumns()) {
			copy(src);
			return;
		}
		ensureAllocateMeta();
		if(_coldata == null) // allocate column data.
			_coldata = new Array[_schema.length];
		synchronized(this) { // make sync locks
			// TODO remove sync locks on array types where they are not needed.
			if(_columnLocks == null) {
				Object[] locks = new Object[_schema.length];
				for(int i = 0; i < locks.length; i++)
					locks[i] = new Object();
				_columnLocks = new SoftReference<>(locks);
			}
		}
		Object[] locks = _columnLocks.get();
		for(int j = cl; j <= cu; j++) { // for each column
			synchronized(locks[j]) { // synchronize on the column.
				_coldata[j] = ArrayFactory.set(_coldata[j], src._coldata[j - cl], rl, ru, _nRow);
			}
		}
	}

	/**
	 * This function will split every Recode map in the column using delimiter Lop.DATATYPE_PREFIX, as Recode map
	 * generated earlier in the form of Code+Lop.DATATYPE_PREFIX+Token and store it in a map which contains token and
	 * code for every unique tokens.
	 *
	 * @param col is the column # from frame data which contains Recode map generated earlier.
	 * @return map of token and code for every element in the input column of a frame containing Recode map
	 */
	public Map<Object, Integer> getRecodeMap(int col) {
		return _coldata[col].getRecodeMap(4);
	}

	@Override
	public FrameBlock merge(FrameBlock that, boolean appendOnly) {
		return merge(that);
	}

	public FrameBlock merge(FrameBlock that) {
		// check for empty input source (nothing to merge)
		if(that == null || that.getNumRows() == 0)
			return this;

		// check dimensions (before potentially copy to prevent implicit dimension change)
		if(getNumRows() != that.getNumRows() || getNumColumns() != that.getNumColumns())
			throw new DMLRuntimeException("Dimension mismatch on merge disjoint (target=" + getNumRows() + "x"
				+ getNumColumns() + ", source=" + that.getNumRows() + "x" + that.getNumColumns() + ")");

		// meta data copy if necessary
		for(int j = 0; j < getNumColumns(); j++)
			if(!that.isColumnMetadataDefault(j)) {
				_colmeta[j].setNumDistinct(that._colmeta[j].getNumDistinct());
				_colmeta[j].setMvValue(that._colmeta[j].getMvValue());
			}

		// core frame block merge through cell copy
		// with column-wide access pattern
		for(int j = 0; j < getNumColumns(); j++) {
			if(_coldata[j].getValueType().equals(that._coldata[j].getValueType()))
				_coldata[j].setNz(that._coldata[j]);
			else
				_coldata[j].setFromOtherTypeNz(that._coldata[j]);
		}
		return this;
	}

	/**
	 * This function ZERO OUT the data in the slicing window applicable for this block.
	 *
	 * @param result         frame block
	 * @param range          index range
	 * @param complementary  ?
	 * @param iRowStartSrc   ?
	 * @param iRowStartDest  ?
	 * @param blen           ?
	 * @param iMaxRowsToCopy ?
	 * @return frame block
	 */
	public FrameBlock zeroOutOperations(FrameBlock result, IndexRange range, boolean complementary, int iRowStartSrc,
		int iRowStartDest, int blen, int iMaxRowsToCopy) {
		int clen = getNumColumns();

		if(result == null)
			result = new FrameBlock(getSchema());
		else {
			result.reset(0, true);
			result.setSchema(getSchema());
		}
		result.ensureAllocatedColumns(blen);

		if(complementary) {
			for(int r = (int) range.rowStart; r <= range.rowEnd && r + iRowStartDest < blen; r++) {
				for(int c = (int) range.colStart; c <= range.colEnd; c++)
					result.set(r + iRowStartDest, c, get(r + iRowStartSrc, c));
			}
		}
		else {
			int r = iRowStartDest;
			for(; r < (int) range.rowStart && r - iRowStartDest < iMaxRowsToCopy; r++)
				for(int c = 0; c < clen; c++/* , offset++ */)
					result.set(r, c, get(r + iRowStartSrc - iRowStartDest, c));

			for(; r <= (int) range.rowEnd && r - iRowStartDest < iMaxRowsToCopy; r++) {
				for(int c = 0; c < (int) range.colStart; c++)
					result.set(r, c, get(r + iRowStartSrc - iRowStartDest, c));

				for(int c = (int) range.colEnd + 1; c < clen; c++)
					result.set(r, c, get(r + iRowStartSrc - iRowStartDest, c));
			}

			for(; r - iRowStartDest < iMaxRowsToCopy; r++)
				for(int c = 0; c < clen; c++)
					result.set(r, c, get(r + iRowStartSrc - iRowStartDest, c));
		}

		return result;
	}

	public FrameBlock getSchemaTypeOf() {
		FrameBlock fb = new FrameBlock(UtilFunctions.nCopies(getNumColumns(), ValueType.STRING));
		fb.appendRow(Arrays.stream(_schema).map(vt -> vt.toString()).toArray(String[]::new));
		return fb;
	}

	public final FrameBlock detectSchema(int k) {
		return FrameLibDetectSchema.detectSchema(this, 0.01, k);
	}

	public final FrameBlock detectSchema(double sampleFraction, int k) {
		return FrameLibDetectSchema.detectSchema(this, sampleFraction, k);
	}

	public final FrameBlock applySchema(FrameBlock schema) {
		return FrameLibApplySchema.applySchema(this, schema);
	}

	public final FrameBlock applySchema(FrameBlock schema, int k) {
		return FrameLibApplySchema.applySchema(this, schema, k);
	}

	/**
	 * Drop the cell value which does not confirms to the data type of its column
	 * 
	 * @param schema of the frame
	 * @return original frame where invalid values are replaced with null
	 */
	public FrameBlock dropInvalidType(FrameBlock schema) {
		// sanity checks
		if(this.getNumColumns() != schema.getNumColumns())
			throw new DMLException("mismatch in number of columns in frame and its schema " + this.getNumColumns()
				+ " != " + schema.getNumColumns());

		// extract the schema in String array
		String[] schemaString = IteratorFactory.getStringRowIterator(schema).next();
		for(int i = 0; i < this.getNumColumns(); i++) {
			Array obj = this.getColumn(i);
			String schemaCol = schemaString[i];
			String type;
			if(schemaCol.contains("FP"))
				type = "FP";
			else if(schemaCol.contains("INT"))
				type = "INT";
			else if(schemaCol.contains("STRING"))
				// In case of String columns, don't do any verification or replacements.
				continue;
			else
				type = schemaCol;

			for(int j = 0; j < this.getNumRows(); j++) {
				if(obj.get(j) == null)
					continue;
				String dataValue = obj.get(j).toString().trim().replace("\"", "").toLowerCase();

				ValueType dataType = FrameUtil.isType(dataValue);

				if(!dataType.toString().contains(type) && !(dataType == ValueType.BOOLEAN && type.equals("INT")) &&
					!(dataType == ValueType.BOOLEAN && type.equals("FP"))) {
					LOG.warn("Datatype detected: " + dataType + " where expected: " + schemaString[i] + " col: "
						+ (i + 1) + ", row:" + (j + 1));

					this.set(j, i, null);
				}
			}
		}
		return this;
	}

	/**
	 * This method validates the frame data against an attribute length constrain if data value in any cell is greater
	 * than the specified threshold of that attribute the output frame will store a null on that cell position, thus
	 * removing the length-violating values.
	 *
	 * @param feaLen vector of valid lengths
	 * @return FrameBlock with invalid values converted into missing values (null)
	 */
	public FrameBlock invalidByLength(MatrixBlock feaLen) {
		// sanity checks
		if(this.getNumColumns() != feaLen.getNumColumns())
			throw new DMLException("mismatch in number of columns in frame and corresponding feature-length vector");

		FrameBlock outBlock = new FrameBlock(this);
		for(int i = 0; i < this.getNumColumns(); i++) {
			if(feaLen.get(0, i) == -1)
				continue;
			int validLength = (int) feaLen.get(0, i);
			Array obj = this.getColumn(i);
			for(int j = 0; j < obj.size(); j++) {
				if(obj.get(j) == null)
					continue;
				String dataValue = obj.get(j).toString();
				if(dataValue.length() > validLength)
					outBlock.set(j, i, null);
			}
		}

		return outBlock;
	}

	public void mapInplace(Function<String, String> fun) {
		for(int j = 0; j < getNumColumns(); j++)
			for(int i = 0; i < getNumRows(); i++) {
				Object tmp = get(i, j);
				set(i, j, (tmp == null) ? tmp : UtilFunctions.objectToObject(_schema[j], fun.apply(tmp.toString())));
			}
	}

	public FrameBlock map(String lambdaExpr, long margin) {
		if(!lambdaExpr.contains("->")) {
			String args = lambdaExpr.substring(lambdaExpr.indexOf('(') + 1, lambdaExpr.indexOf(')'));
			if(args.contains(",")) {
				String[] arguments = args.split(",");
				return DMVUtils.syntacticalPatternDiscovery(this, Double.parseDouble(arguments[0]), arguments[1]);
			}
			else if(args.contains(";")) {
				String[] arguments = args.split(";");
				return EMAUtils.exponentialMovingAverageImputation(this, Integer.parseInt(arguments[0]), arguments[1],
					Integer.parseInt(arguments[2]), Double.parseDouble(arguments[3]), Double.parseDouble(arguments[4]),
					Double.parseDouble(arguments[5]));
			}
		}
		if(lambdaExpr.contains("jaccardSim"))
			return mapDist(getCompiledFunction(lambdaExpr, margin));
		return map(getCompiledFunction(lambdaExpr, margin), margin);
	}

	public FrameBlock frameRowReplication(FrameBlock rowToreplicate) {
		FrameBlock out = new FrameBlock(this);
		if(this.getNumColumns() != rowToreplicate.getNumColumns())
			throw new DMLRuntimeException("Mismatch number of columns");
		if(rowToreplicate.getNumRows() > 1)
			throw new DMLRuntimeException("only supported single rows frames to replicate");
		for(int i = 0; i < this.getNumRows(); i++)
			for(int j = 0; j < this.getNumColumns(); j++)
				out.set(i, j, rowToreplicate.get(0, j));
		return out;
	}

	@SuppressWarnings("deprecation")
	public FrameBlock valueSwap(FrameBlock schema) {
		String[] schemaString = IteratorFactory.getStringRowIterator(schema).next();
		String dataValue2 = null;
		double minSimScore = 0;
		int bestIdx = 0;
		// remove the precision info
		for(int i = 0; i < schemaString.length; i++)
			schemaString[i] = schemaString[i].replaceAll("\\d", "");

		double[] minColLength = new double[this.getNumColumns()];
		double[] maxColLength = new double[this.getNumColumns()];

		for(int k = 0; k < this.getNumColumns(); k++) {
			Pair<Integer, Integer> minMax = _coldata[k].getMinMaxLength();
			maxColLength[k] = minMax.getKey();
			minColLength[k] = minMax.getValue();
		}

		ArrayList<Integer> probColList = new ArrayList();
		for(int i = 0; i < this.getNumColumns(); i++) {
			for(int j = 0; j < this.getNumRows(); j++) {
				if(this.get(j, i) == null)
					continue;
				String dataValue = this.get(j, i).toString().trim().replace("\"", "").toLowerCase();
				ValueType dataType = FrameUtil.isType(dataValue);

				String type = dataType.toString().replaceAll("\\d", "");
				// get the avergae column length
				if(!dataType.toString().contains(schemaString[i]) &&
					!(dataType == ValueType.BOOLEAN && schemaString[i].equals("INT")) &&
					!(dataType == ValueType.BOOLEAN && schemaString[i].equals("FP")) &&
					!(dataType.toString().contains("INT") && schemaString[i].equals("FP"))) {
					LOG.warn("conflict " + dataType + " " + schemaString[i] + " " + dataValue);
					// check the other column with satisfy the data type of this value
					for(int w = 0; w < schemaString.length; w++) {
						if(schemaString[w].equals(type) && dataValue.length() > minColLength[w] &&
							dataValue.length() < maxColLength[w] && (w != i)) {
							Object item = this.get(j, w);
							String dataValueProb = (item != null) ? item.toString().trim().replace("\"", "")
								.toLowerCase() : "0";
							ValueType dataTypeProb = FrameUtil.isType(dataValueProb);
							if(!dataTypeProb.toString().equals(schemaString[w])) {
								bestIdx = w;
								break;
							}
							probColList.add(w);
						}
					}
					// if we have more than one column that is the probable match for this value then find the most
					// appropriate one by using the similarity score
					if(probColList.size() > 1) {
						for(int w : probColList) {
							int randomIndex = ThreadLocalRandom.current().nextInt(0, getNumRows() - 1);
							Object value = this.get(randomIndex, w);
							if(value != null) {
								dataValue2 = value.toString();
							}

							// compute distance between sample and invalid value
							double simScore = 0;
							if(!(dataValue == null) && !(dataValue2 == null))
								simScore = StringUtils.getLevenshteinDistance(dataValue, dataValue2);
							if(simScore < minSimScore) {
								minSimScore = simScore;
								bestIdx = w;
							}
						}
					}
					else if(probColList.size() > 0) {
						bestIdx = probColList.get(0);
					}
					String tmp = dataValue;
					this.set(j, i, this.get(j, bestIdx));
					this.set(j, bestIdx, tmp);
				}
			}
		}
		return this;
	}

	public FrameBlock map(FrameMapFunction lambdaExpr, long margin) {
		// Prepare temporary output array
		String[][] output = new String[getNumRows()][getNumColumns()];

		if(margin == 1) {
			// Execute map function on rows
			for(int i = 0; i < getNumRows(); i++) {
				String[] row = new String[getNumColumns()];
				for(int j = 0; j < getNumColumns(); j++) {
					Array input = getColumn(j);
					row[j] = String.valueOf(input.get(i));
				}
				output[i] = lambdaExpr.apply(row);
			}
		}
		else if(margin == 2) {
			// Execute map function on columns
			for(int j = 0; j < getNumColumns(); j++) {
				// since more rows can be allocated, mutable array
				String[] actualColumn = Arrays.copyOfRange((String[]) getColumnData(j), 0, getNumRows());
				String[] outColumn = lambdaExpr.apply(actualColumn);
				for(int i = 0; i < getNumRows(); i++)
					output[i][j] = outColumn[i];
			}
		}
		else {
			// Execute map function on all cells
			for(int j = 0; j < getNumColumns(); j++) {
				Array input = getColumn(j);
				for(int i = 0; i < input.size(); i++)
					if(input.get(i) != null)
						output[i][j] = lambdaExpr.apply(String.valueOf(input.get(i)));
			}
		}
		return new FrameBlock(UtilFunctions.nCopies(getNumColumns(), ValueType.STRING), output);
	}

	public FrameBlock mapDist(FrameMapFunction lambdaExpr) {
		String[][] output = new String[getNumRows()][getNumRows()];
		for(String[] row : output)
			Arrays.fill(row, "0.0");
		Array input = getColumn(0);
		for(int j = 0; j < input.size() - 1; j++) {
			for(int i = j + 1; i < input.size(); i++)
				if(input.get(i) != null && input.get(j) != null) {
					output[j][i] = lambdaExpr.apply(String.valueOf(input.get(j)), String.valueOf(input.get(i)));
				}
		}
		return new FrameBlock(UtilFunctions.nCopies(getNumRows(), ValueType.STRING), output);
	}

	public static FrameMapFunction getCompiledFunction(String lambdaExpr, long margin) {
		String cname = "StringProcessing" + CLASS_ID.getNextID();
		StringBuilder sb = new StringBuilder();
		String[] parts = lambdaExpr.split("->");
		if(parts.length != 2)
			throw new DMLRuntimeException("Unsupported lambda expression: " + lambdaExpr);
		String[] varname = parts[0].replaceAll("[()]", "").split(",");
		String expr = parts[1].trim();

		// construct class code
		sb.append("import org.apache.sysds.runtime.util.UtilFunctions;\n");
		sb.append("import org.apache.sysds.runtime.util.PorterStemmer;\n");
		sb.append("import org.apache.sysds.runtime.frame.data.FrameBlock.FrameMapFunction;\n");
		sb.append("import java.util.Arrays;\n");
		sb.append("public class " + cname + " extends FrameMapFunction {\n");
		if(margin != 0) {
			sb.append("public String[] apply(String[] " + varname[0].trim() + ") {\n");
			sb.append("  return UtilFunctions.toStringArray(" + expr + "); }}\n");
		}
		else {
			if(varname.length == 1) {
				sb.append("public String apply(String " + varname[0].trim() + ") {\n");
				sb.append("  return String.valueOf(" + expr + "); }}\n");
			}
			else if(varname.length == 2) {
				sb.append(
					"public String apply(String " + varname[0].trim() + ", String " + varname[1].trim() + ") {\n");
				sb.append("  return String.valueOf(" + expr + "); }}\n");
			}
		}
		// compile class, and create FrameMapFunction object
		try {
			return (FrameMapFunction) CodegenUtils.compileClass(cname, sb.toString()).getDeclaredConstructor()
				.newInstance();
		}
		catch(InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
			| NoSuchMethodException | SecurityException e) {
			throw new DMLRuntimeException("Failed to compile FrameMapFunction.", e);
		}
	}

	public static class FrameMapFunction implements Serializable {
		private static final long serialVersionUID = -8398572153616520873L;

		public String apply(String input) {
			return null;
		}

		public String apply(String input1, String input2) {
			return null;
		}

		public String[] apply(String[] input1) {
			return null;
		}
	}

	public <T> FrameBlock replaceOperations(String pattern, String replacement) {
		FrameBlock ret = new FrameBlock(this);

		boolean NaNp = "NaN".equals(pattern);
		boolean NaNr = "NaN".equals(replacement);
		ValueType patternType = UtilFunctions
			.isBoolean(pattern) ? ValueType.BOOLEAN : (NumberUtils.isCreatable(pattern) |
				NaNp ? (UtilFunctions.isIntegerNumber(pattern) ? ValueType.INT64 : ValueType.FP64) : ValueType.STRING);
		ValueType replacementType = UtilFunctions.isBoolean(replacement) ? ValueType.BOOLEAN : (NumberUtils
			.isCreatable(replacement) |
			NaNr ? (UtilFunctions.isIntegerNumber(replacement) ? ValueType.INT64 : ValueType.FP64) : ValueType.STRING);

		if(patternType != replacementType || !ValueType.isSameTypeString(patternType, replacementType))
			throw new DMLRuntimeException(
				"Pattern and replacement types should be same: " + patternType + " " + replacementType);

		for(int i = 0; i < ret.getNumColumns(); i++) {
			Array colData = ret._coldata[i];
			for(int j = 0;
				j < colData.size() &&
					(ValueType.isSameTypeString(_schema[i], patternType) || _schema[i] == ValueType.STRING);
				j++) {
				T patternNew = (T) UtilFunctions.stringToObject(_schema[i], pattern);
				T replacementNew = (T) UtilFunctions.stringToObject(_schema[i], replacement);

				Object ent = colData.get(j);
				if(ent != null && ent.toString().equals(patternNew.toString()))
					colData.set(j, replacementNew);
				else if(ent instanceof String && ent.equals(pattern))
					colData.set(j, replacement);
			}
		}
		return ret;
	}

	public FrameBlock removeEmptyOperations(boolean rows, boolean emptyReturn, MatrixBlock select) {
		return FrameLibRemoveEmpty.removeEmpty(this, rows, emptyReturn, select);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("FrameBlock");
		if(_colnames != null) {
			sb.append("\n");
			sb.append(Arrays.toString(_colnames));
		}
		if(!isColumnMetadataDefault()) {
			sb.append("\n");
			sb.append(Arrays.toString(_colmeta));
		}
		sb.append("\n");
		sb.append(Arrays.toString(_schema));
		sb.append("\n");
		if(_coldata != null) {
			for(int i = 0; i < _coldata.length; i++) {
				sb.append(_coldata[i]);
				sb.append("\n");
			}
		}
		return sb.toString();
	}
}
