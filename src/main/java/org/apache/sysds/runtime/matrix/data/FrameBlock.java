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

package org.apache.sysds.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;
import java.lang.ref.SoftReference;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.math.NumberUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.ValueComparisonFunction;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DMVUtils;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.EMAUtils;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;

@SuppressWarnings({"rawtypes","unchecked"}) //allow generic native arrays
public class FrameBlock implements CacheBlock, Externalizable {
	private static final long serialVersionUID = -3993450030207130665L;
	private static final Log LOG = LogFactory.getLog(FrameBlock.class.getName());
	private static final IDSequence CLASS_ID = new IDSequence();

	public static final int BUFFER_SIZE = 1 * 1000 * 1000; //1M elements, size of default matrix block

	//internal configuration
	private static final boolean REUSE_RECODE_MAPS = true;

	/** The number of rows of the FrameBlock */
	private int _numRows = -1;

	/** The schema of the data frame as an ordered list of value types */
	private ValueType[] _schema = null;

	/** The column names of the data frame as an ordered list of strings, allocated on-demand */
	private String[] _colnames = null;

	private ColumnMetadata[] _colmeta = null;

	/** The data frame data as an ordered list of columns */
	private Array[] _coldata = null;

	/** Cached size in memory to avoid repeated scans of string columns */
	long _msize = -1;

	public FrameBlock() {
		_numRows = 0;
	}

	/**
	 * Copy constructor for frame blocks, which uses a shallow copy for
	 * the schema (column types and names) but a deep copy for meta data
	 * and actual column data.
	 *
	 * @param that frame block
	 */
	public FrameBlock(FrameBlock that) {
		this(that.getSchema(), that.getColumnNames(false));
		copy(that);
		setColumnMetadata(that.getColumnMetadata());
	}

	public FrameBlock(int ncols, ValueType vt) {
		this();
		_schema = UtilFunctions.nCopies(ncols, vt);
		_colnames = null; //default not materialized
		_colmeta = new ColumnMetadata[ncols];
		for( int j=0; j<ncols; j++ )
			_colmeta[j] = new ColumnMetadata(0);
	}

	public FrameBlock(ValueType[] schema) {
		this(schema, new String[0][]);
	}

	public FrameBlock(ValueType[] schema, String[] names) {
		this(schema, names, new String[0][]);
	}

	public FrameBlock(ValueType[] schema, String[][] data) {
		//default column names not materialized
		this(schema, null, data);

	}

	public FrameBlock(ValueType[] schema, String[] names, String[][] data) {
		_numRows = 0; //maintained on append
		_schema = schema;
		_colnames = names;
		_colmeta = new ColumnMetadata[_schema.length];
		for( int j=0; j<_schema.length; j++ )
			_colmeta[j] = new ColumnMetadata(0);
		for( int i=0; i<data.length; i++ )
			appendRow(data[i]);
	}

	/**
	 * Get the number of rows of the frame block.
	 *
	 * @return number of rows
	 */
	@Override
	public int getNumRows() {
		return _numRows;
	}

	@Override
	public double getDouble(int r, int c) {
		Object o = get(r, c);
		if(o == null || (getSchema()[c] == ValueType.STRING && o.toString().isEmpty()))
			return 0;
		return UtilFunctions.objectToDouble(getSchema()[c], o);
	}

	@Override
	public double getDoubleNaN(int r, int c) {
		Object o = get(r, c);
		if(o == null || (getSchema()[c] == ValueType.STRING && o.toString().isEmpty()))
			return Double.NaN;
		return UtilFunctions.objectToDouble(getSchema()[c], o);

	}

	@Override
	public String getString(int r, int c) {
		Object o = get(r, c);
		String s = (o == null) ? null : o.toString();
		if(s != null && s.isEmpty())
			return null;
		return s;
	}

	public void setNumRows(int numRows) {
		_numRows = numRows;
	}

	/**
	 * Get the number of columns of the frame block, that is
	 * the number of columns defined in the schema.
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
	 * Returns the column names of the frame block. This method
	 * allocates default column names if required.
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
	 * Returns the column names of the frame block. This method
	 * allocates default column names if required.
	 *
	 * @param alloc if true, create column names
	 * @return array of column names
	 */
	public String[] getColumnNames(boolean alloc) {
		if( _colnames == null && alloc )
			_colnames = createColNames(getNumColumns());
		return _colnames;
	}

	/**
	 * Returns the column name for the requested column. This
	 * method allocates default column names if required.
	 *
	 * @param c column index
	 * @return column name
	 */
	public String getColumnName(int c) {
		if( _colnames == null )
			_colnames = createColNames(getNumColumns());
		return _colnames[c];
	}

	public void setColumnNames(String[] colnames) {
		_colnames = colnames;
	}

	public ColumnMetadata[] getColumnMetadata() {
		return _colmeta;
	}

	public ColumnMetadata getColumnMetadata(int c) {
		return _colmeta[c];
	}

	public boolean isColumnMetadataDefault() {
		boolean ret = true;
		for( int j=0; j<getNumColumns() && ret; j++ )
			ret &= isColumnMetadataDefault(j);
		return ret;
	}

	public boolean isColumnMetadataDefault(int c) {
		return _colmeta[c].getMvValue() == null
			&& _colmeta[c].getNumDistinct() == 0;
	}

	public void setColumnMetadata(ColumnMetadata[] colmeta) {
		System.arraycopy(colmeta, 0, _colmeta, 0, _colmeta.length);
	}

	public void setColumnMetadata(int c, ColumnMetadata colmeta) {
		_colmeta[c] = colmeta;
	}

	/**
	 * Creates a mapping from column names to column IDs, i.e.,
	 * 1-based column indexes
	 *
	 * @return map of column name keys and id values
	 */
	public Map<String,Integer> getColumnNameIDMap() {
		Map<String, Integer> ret = new HashMap<>();
		for( int j=0; j<getNumColumns(); j++ )
			ret.put(getColumnName(j), j+1);
		return ret;
	}

	/**
	 * Allocate column data structures if necessary, i.e., if schema specified
	 * but not all column data structures created yet.
	 *
	 * @param numRows number of rows
	 */
	public void ensureAllocatedColumns(int numRows) {
		_msize = -1;
		//early abort if already allocated
		if( _coldata != null && _schema.length == _coldata.length ) {
			//handle special case that to few rows allocated
			if( _numRows < numRows ) {
				String[] tmp = new String[getNumColumns()];
				int len = numRows - _numRows;
				for(int i=0; i<len; i++)
					appendRow(tmp);
			}
			return;
		}
		//allocate column meta data if necessary
		if( _colmeta == null || _schema.length != _colmeta.length ) {
			_colmeta = new ColumnMetadata[_schema.length];
			for( int j=0; j<_schema.length; j++ )
				_colmeta[j] = new ColumnMetadata(0);
		}
		//allocate columns if necessary
		_coldata = new Array[_schema.length];
		for( int j=0; j<_schema.length; j++ ) {
			switch( _schema[j] ) {
				case STRING:  _coldata[j] = new StringArray(new String[numRows]); break;
				case BOOLEAN: _coldata[j] = new BooleanArray(new boolean[numRows]); break;
				case INT32:   _coldata[j] = new IntegerArray(new int[numRows]); break;
				case INT64:   _coldata[j] = new LongArray(new long[numRows]); break;
				case FP32:   _coldata[j] = new FloatArray(new float[numRows]); break;
				case FP64:   _coldata[j] = new DoubleArray(new double[numRows]); break;
				default: throw new RuntimeException("Unsupported value type: "+_schema[j]);
			}
		}
		_numRows = numRows;
	}

	/**
	 * Checks for matching column sizes in case of existing columns.
	 *
	 * @param newlen number of rows to compare with existing number of rows
	 */
	public void ensureColumnCompatibility(int newlen) {
		if( _coldata!=null && _coldata.length > 0 && _numRows != newlen )
			throw new RuntimeException("Mismatch in number of rows: "+newlen+" (expected: "+_numRows+")");
	}

	public static String[] createColNames(int size) {
		return createColNames(0, size);
	}

	public static String[] createColNames(int off, int size) {
		String[] ret = new String[size];
		for( int i=off+1; i<=off+size; i++ )
			ret[i-off-1] = createColName(i);
		return ret;
	}

	public static String createColName(int i) {
		return "C" + i;
	}

	public boolean isColNamesDefault() {
		boolean ret = (_colnames != null);
		for( int j=0; j<getNumColumns() && ret; j++ )
			ret &= isColNameDefault(j);
		return ret;
	}

	public boolean isColNameDefault(int i) {
		return _colnames==null
			|| _colnames[i].equals("C"+(i+1));
	}

	public void recomputeColumnCardinality() {
		for( int j=0; j<getNumColumns(); j++ ) {
			int card = 0;
			for( int i=0; i<getNumRows(); i++ )
				card += (get(i, j) != null) ? 1 : 0;
			_colmeta[j].setNumDistinct(card);
		}
	}

	///////
	// basic get and set functionality

	/**
	 * Gets a boxed object of the value in position (r,c).
	 *
	 * @param r	row index, 0-based
	 * @param c	column index, 0-based
	 * @return object of the value at specified position
	 */
	public Object get(int r, int c) {
		return _coldata[c].get(r);
	}

	/**
	 * Sets the value in position (r,c), where the input is assumed
	 * to be a boxed object consistent with the schema definition.
	 *
	 * @param r row index
	 * @param c column index
	 * @param val value to set at specified position
	 */
	public void set(int r, int c, Object val) {
		_coldata[c].set(r, UtilFunctions.objectToObject(_schema[c], val));
		_msize = -1;
	}

	public void reset(int nrow, boolean clearMeta) {
		if( clearMeta ) {
			_schema = null;
			_colnames = null;
			if( _colmeta != null ) {
				for( int i=0; i<_colmeta.length; i++ )
					if( !isColumnMetadataDefault(i) )
						_colmeta[i] = new ColumnMetadata(0);
			}
		}
		if(_coldata != null) {
			for( int i=0; i < _coldata.length; i++ )
				_coldata[i].reset(nrow);
		}
		_msize = -1;
	}

	public void reset() {
		reset(0, true);
	}


	/**
	 * Append a row to the end of the data frame, where all row fields
	 * are boxed objects according to the schema.
	 *
	 * @param row array of objects
	 */
	public void appendRow(Object[] row) {
		ensureAllocatedColumns(0);
		for( int j=0; j<row.length; j++ )
			_coldata[j].append(row[j]);
		_numRows++;
	}

	/**
	 * Append a row to the end of the data frame, where all row fields
	 * are string encoded.
	 *
	 * @param row array of strings
	 */
	public void appendRow(String[] row) {
		ensureAllocatedColumns(0);
		for( int j=0; j<row.length; j++ )
			_coldata[j].append(row[j]);
		_numRows++;
	}

	/**
	 * Append a column of value type STRING as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of strings
	 */
	public void appendColumn(String[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.STRING);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new StringArray(col)} :
			(Array[]) ArrayUtils.add(_coldata, new StringArray(col));
		_numRows = col.length;
		_msize = -1;
	}

	/**
	 * Append a column of value type BOOLEAN as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of booleans
	 */
	public void appendColumn(boolean[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.BOOLEAN);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new BooleanArray(col)} :
			(Array[]) ArrayUtils.add(_coldata, new BooleanArray(col));
		_numRows = col.length;
		_msize = -1;
	}

	/**
	 * Append a column of value type INT as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of longs
	 */
	public void appendColumn(int[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.INT32);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new IntegerArray(col)} :
			(Array[]) ArrayUtils.add(_coldata, new IntegerArray(col));
		_numRows = col.length;
		_msize = -1;
	}
	/**
	 * Append a column of value type LONG as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of longs
	 */
	public void appendColumn(long[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.INT64);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new LongArray(col)} :
			(Array[]) ArrayUtils.add(_coldata, new LongArray(col));
		_numRows = col.length;
		_msize = -1;
	}

	/**
	 * Append a column of value type float as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of doubles
	 */
	public void appendColumn(float[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.FP32);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new FloatArray(col)} :
				(Array[]) ArrayUtils.add(_coldata, new FloatArray(col));
		_numRows = col.length;
		_msize = -1;
	}
	/**
	 * Append a column of value type DOUBLE as the last column of
	 * the data frame. The given array is wrapped but not copied
	 * and hence might be updated in the future.
	 *
	 * @param col array of doubles
	 */
	public void appendColumn(double[] col) {
		ensureColumnCompatibility(col.length);
		String[] colnames = getColumnNames(); //before schema modification
		_schema = (ValueType[]) ArrayUtils.add(_schema, ValueType.FP64);
		_colnames = (String[]) ArrayUtils.add(colnames, createColName(_schema.length));
		_coldata = (_coldata==null) ? new Array[]{new DoubleArray(col)} :
			(Array[]) ArrayUtils.add(_coldata, new DoubleArray(col));
		_numRows = col.length;
		_msize = -1;
	}

	/**
	 * Append a set of column of value type DOUBLE at the end of the frame
	 * in order to avoid repeated allocation with appendColumns. The given
	 * array is wrapped but not copied and hence might be updated in the future.
	 *
	 * @param cols 2d array of doubles
	 */
	public void appendColumns(double[][] cols) {
		int ncol = cols.length;
		boolean empty = (_schema == null);
		ValueType[] tmpSchema = UtilFunctions.nCopies(ncol, ValueType.FP64);
		Array[] tmpData = new Array[ncol];
		for( int j=0; j<ncol; j++ )
			tmpData[j] = new DoubleArray(cols[j]);
		_colnames = empty ? null : (String[]) ArrayUtils.addAll(getColumnNames(),
			createColNames(getNumColumns(), ncol)); //before schema modification
		_schema = empty ? tmpSchema : (ValueType[]) ArrayUtils.addAll(_schema, tmpSchema);
		_coldata = empty ? tmpData : (Array[]) ArrayUtils.addAll(_coldata, tmpData);
		_numRows = cols[0].length;
		_msize = -1;
	}

	public void appendColumn(ValueType vt, Array col) {
		switch (vt) {
			case STRING:
				appendColumn(((StringArray) col).get());
				break;
			case BOOLEAN:
				appendColumn(((BooleanArray) col).get());
				break;
			case INT32:
				appendColumn(((IntegerArray) col).get());
				break;
			case INT64:
				appendColumn(((LongArray) col).get());
				break;
			case FP32:
				appendColumn(((FloatArray) col).get());
				break;
			case FP64:
				appendColumn(((DoubleArray) col).get());
				break;
			default:
				throw new RuntimeException("Unsupported value type: " + vt);
		}
	}

	public Object getColumnData(int c) {
		switch(_schema[c]) {
			case STRING:  return ((StringArray)_coldata[c])._data;
			case BOOLEAN: return ((BooleanArray)_coldata[c])._data;
			case INT64:   return ((LongArray)_coldata[c])._data;
			case INT32:   return ((IntegerArray)_coldata[c])._data;
			case FP64:    return ((DoubleArray)_coldata[c])._data;
			default:      return null;
	 	}
	}

	public String getColumnType(int c){
		switch(_schema[c]) {
			case STRING:  return "String";
			case BOOLEAN: return "Boolean";
			case INT64:   return "Long";
			case INT32:   return "Int";
			case FP64:    return "Double";
			default:      return null;
	 	}
	}

	/**
	 * Get a specific index as bytes, this method is used to parse the strings into Python.
	 * It should only be used in columns where the datatype is String.
	 * Since in other cases it might be faster to return other types.
	 * 
	 * Note that P 
	 *
	 * @param c The column index.
	 * @param r The row index.
	 * @return The returned byte array.
	 */
	public byte[] getIndexAsBytes(int c, int r){
		switch(_schema[c]){
			case STRING:
				String[] col = ((StringArray)_coldata[c])._data;
				if(col[r] != null)
					return col[r].getBytes();
				else
					return null;
			default:
				throw new NotImplementedException();
		}
	}

	public byte[] getColumnAsBytes(int c){
		final int nRow = getNumRows();
		switch(_schema[c]){
			case INT64:
				long[] colLong = ((LongArray)_coldata[c])._data;
				ByteBuffer longBuffer = ByteBuffer.allocate(8 * nRow);
				longBuffer.order(ByteOrder.LITTLE_ENDIAN);
				for(int i = 0; i <  nRow; i++)
					longBuffer.putLong(colLong[i]);
				return longBuffer.array();
			case INT32:
				int[] colInt = ((IntegerArray)_coldata[c])._data;
				ByteBuffer intBuffer = ByteBuffer.allocate(4 *  nRow);
				intBuffer.order(ByteOrder.LITTLE_ENDIAN);
				for(int i = 0; i < nRow; i++)
					intBuffer.putInt(colInt[i]);
				return intBuffer.array();
			case FP64:
				double[] colDouble = ((DoubleArray)_coldata[c])._data;
				ByteBuffer doubleBuffer = ByteBuffer.allocate(8 * nRow);
				doubleBuffer.order(ByteOrder.nativeOrder());
				for(int i = 0; i < nRow; i++)
					doubleBuffer.putDouble(colDouble[i]);
				return doubleBuffer.array();
			case FP32:
				float[] colFloat = ((FloatArray)_coldata[c])._data;
				ByteBuffer floatBuffer = ByteBuffer.allocate(8 * nRow);
				floatBuffer.order(ByteOrder.nativeOrder());
				for(int i = 0; i < nRow; i++)
					floatBuffer.putDouble(colFloat[i]);
				return floatBuffer.array();
			case BOOLEAN:
				boolean[] colBool = ((BooleanArray)_coldata[c])._data;
				// over allocating here.. we could maybe bit pack?
				ByteBuffer booleanBuffer = ByteBuffer.allocate(nRow);
				booleanBuffer.order(ByteOrder.nativeOrder());
				for(int i = 0; i < nRow; i++)
					booleanBuffer.put((byte)(colBool[i]? 1:0));
				return booleanBuffer.array();
			default:
				throw new NotImplementedException();
		}
	}

	public Array getColumn(int c) {
		return _coldata[c];
	}

	public void setColumn(int c, Array column) {
		if( _coldata == null )
			_coldata = new Array[getNumColumns()];
		_coldata[c] = column;
		_msize = -1;
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as strings independent of their value types.
	 *
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator() {
		return new StringRowIterator(0, _numRows);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are
	 * encoded as strings independent of their value types.
	 *
	 * @param cols column selection, 1-based
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator(int[] cols) {
		return new StringRowIterator(0, _numRows, cols);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are encoded as strings independent of their value
	 * types.
	 *
	 * @param colID column selection, 1-based
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator(int colID) {
		return new StringRowIterator(0, _numRows, new int[] {colID});
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as strings independent of their value types.
	 *
	 * @param rl lower row index
	 * @param ru upper row index
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator(int rl, int ru) {
		return new StringRowIterator(rl, ru);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are
	 * encoded as strings independent of their value types.
	 *
	 * @param rl lower row index
	 * @param ru upper row index
	 * @param cols column selection, 1-based
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator(int rl, int ru, int[] cols) {
		return new StringRowIterator(rl, ru, cols);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are
	 * encoded as strings independent of their value types.
	 *
	 * @param rl lower row index
	 * @param ru upper row index
	 * @param colID columnID, 1-based
	 * @return string array iterator
	 */
	public Iterator<String[]> getStringRowIterator(int rl, int ru, int colID) {
		return new StringRowIterator(rl, ru, new int[] {colID});
	}


	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as boxed objects according to their value types.
	 *
	 * @return object array iterator
	 */
	public Iterator<Object[]> getObjectRowIterator() {
		return new ObjectRowIterator(0, _numRows);
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as boxed objects according to the value types of the provided
	 * target schema.
	 *
	 * @param schema target schema of objects
	 * @return object array iterator
	 */
	public Iterator<Object[]> getObjectRowIterator(ValueType[] schema) {
		ObjectRowIterator iter = new ObjectRowIterator(0, _numRows);
		iter.setSchema(schema);
		return iter;
	}

	/**
	 * Get a row iterator over the frame where all selected fields are
	 * encoded as boxed objects according to their value types.
	 *
	 * @param cols column selection, 1-based
	 * @return object array iterator
	 */
	public Iterator<Object[]> getObjectRowIterator(int[] cols) {
		return new ObjectRowIterator(0, _numRows, cols);
	}

	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as boxed objects according to their value types.
	 *
	 * @param rl lower row index
	 * @param ru upper row index
	 * @return object array iterator
	 */
	public Iterator<Object[]> getObjectRowIterator(int rl, int ru) {
		return new ObjectRowIterator(rl, ru);
	}

	/**
	 * Get a row iterator over the frame where all selected fields are
	 * encoded as boxed objects according to their value types.
	 *
	 * @param rl lower row index
	 * @param ru upper row index
	 * @param cols column selection, 1-based
	 * @return object array iterator
	 */
	public Iterator<Object[]> getObjectRowIterator(int rl, int ru, int[] cols) {
		return new ObjectRowIterator(rl, ru, cols);
	}

	///////
	// serialization / deserialization (implementation of writable and externalizable)
	// FIXME for FrameBlock fix write and readFields, it does not work if the Arrays are not yet
	// allocated (after fixing remove hack in FederatedWorkerHandler.createFrameEncodeMeta(FederatedRequest) call to
	// FrameBlock.ensureAllocatedColumns())

	@Override
	public void write(DataOutput out) throws IOException {
		boolean isDefaultMeta = isColNamesDefault()
				&& isColumnMetadataDefault();
		//write header (rows, cols, default)
		out.writeInt(getNumRows());
		out.writeInt(getNumColumns());
		out.writeBoolean(isDefaultMeta);
		//write columns (value type, data)
		for( int j=0; j<getNumColumns(); j++ ) {
			byte type = (byte)_schema[j].ordinal();
			if( _coldata == null || _coldata[j] == null )
				type *= -1; //negative to indicate non-existence
			out.writeByte(type);
			if( !isDefaultMeta ) {
				out.writeUTF(getColumnName(j));
				out.writeLong(_colmeta[j].getNumDistinct());
				out.writeUTF( (_colmeta[j].getMvValue()!=null) ?
					_colmeta[j].getMvValue() : "" );
			}
			if( type >= 0 )
				_coldata[j].write(out);
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		//read head (rows, cols)
		_numRows = in.readInt();
		int numCols = in.readInt();
		boolean isDefaultMeta = in.readBoolean();
		//allocate schema/meta data arrays
		_schema = (_schema!=null && _schema.length==numCols) ?
			_schema : new ValueType[numCols];
		_colnames = (_colnames != null && _colnames.length==numCols) ?
			_colnames : new String[numCols];
		_colmeta = (_colmeta != null && _colmeta.length==numCols) ?
			_colmeta : new ColumnMetadata[numCols];
		_coldata = (_coldata!=null && _coldata.length==numCols) ?
			_coldata : new Array[numCols];
		//read columns (value type, meta, data)
		for( int j=0; j<numCols; j++ ) {
			byte type = in.readByte();
			ValueType vt = ValueType.values()[Math.abs(type)];
			String name = isDefaultMeta ? createColName(j) : in.readUTF();
			long ndistinct = isDefaultMeta ? 0 : in.readLong();
			String mvvalue = isDefaultMeta ? null : in.readUTF();
			Array arr = null;
			if( type > 0 ) { //non-empty column
				switch( vt ) {
					case STRING:  arr = new StringArray(new String[_numRows]); break;
					case BOOLEAN: arr = new BooleanArray(new boolean[_numRows]); break;
					case INT64:     arr = new LongArray(new long[_numRows]); break;
					case FP64:  arr = new DoubleArray(new double[_numRows]); break;
					case INT32: arr = new IntegerArray(new int[_numRows]); break;
					case FP32:  arr = new FloatArray(new float[_numRows]); break;
					default: throw new IOException("Unsupported value type: "+vt);
				}
				arr.readFields(in);
			}
			_schema[j] = vt;
			_colnames[j] = name;
			_colmeta[j] = new ColumnMetadata(ndistinct,
				(mvvalue==null || mvvalue.isEmpty()) ? null : mvvalue);
			_coldata[j] = arr;
		}
		_msize = -1;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		//redirect serialization to writable impl
		write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException {
		//redirect deserialization to writable impl
		readFields(in);
	}

	////////
	// CacheBlock implementation

	@Override
	public long getInMemorySize() {
		//reuse previously computed size
		if( _msize > 0 )
			return _msize;

		//frame block header
		long size = 16 + 4; //object, num rows

		//schema array (overhead and int entries)
		int clen = getNumColumns();
		size += 8 + 32 + clen * 4;

		//colname array (overhead and string entries)
		size += 8 + ((_colnames!=null) ? 32 : 0);
		for( int j=0; j<clen && _colnames!=null; j++ )
			size += getInMemoryStringSize(getColumnName(j));

		//meta data array (overhead and entries)
		size += 8 + 32;
		for( int j=0; j<clen; j++ ) {
			size += 16 + 8 + 8 //object, long num distinct, ref mv
				+ getInMemoryStringSize(_colmeta[j].getMvValue());
		}

		//data array (overhead and entries)
		size += 8 + 32 + clen * (16+4+8+32);
		for( int j=0; j<clen; j++ ) {
			switch( _schema[j] ) {
				case BOOLEAN: size += _numRows; break;
				case INT64:
				case FP64: size += 8*_numRows; break;
				case STRING:
					StringArray arr = (StringArray)_coldata[j];
					for( int i=0; i<_numRows; i++ )
						size += getInMemoryStringSize(arr.get(i));
					break;
				default: //not applicable
			}
		}

		return _msize = size;
	}

	@Override
	public long getExactSerializedSize() {
		//header: 2xint, boolean
		long size = 9;

		//column sizes
		boolean isDefaultMeta = isColNamesDefault()
				&& isColumnMetadataDefault();
		for( int j=0; j<getNumColumns(); j++ ) {
			size += 1; //column schema
			if( !isDefaultMeta ) {
				size += IOUtilFunctions.getUTFSize(getColumnName(j));
				size += 8;
				size += IOUtilFunctions.getUTFSize(_colmeta[j].getMvValue());
			}
			switch( _schema[j] ) {
				case BOOLEAN: size += _numRows; break;
				case INT64:
				case FP64: size += 8*_numRows; break;
				case STRING:
					StringArray arr = (StringArray)_coldata[j];
					for( int i=0; i<_numRows; i++ )
						size += IOUtilFunctions.getUTFSize(arr.get(i));
					break;
				default: //not applicable
			}
		}

		return size;
	}

	@Override
	public boolean isShallowSerialize() {
		return isShallowSerialize(false);
	}

	@Override
	public boolean isShallowSerialize(boolean inclConvert) {
		//shallow serialize if non-string schema because a frame block
		//is always dense but strings have large array overhead per cell
		boolean ret = true;
		for( int j=0; j<_schema.length && ret; j++ )
			ret &= (_schema[j] != ValueType.STRING);
		return ret;
	}

	@Override
	public void toShallowSerializeBlock() {
		//do nothing (not applicable).
	}

	@Override
	public void compactEmptyBlock() {
		//do nothing
	}

	/**
	 * Returns the in-memory size in bytes of the given string value.
	 *
	 * @param value string value
	 * @return in-memory size of string value
	 */
	private static long getInMemoryStringSize(String value) {
		if( value == null )
			return 0;
		return 16 + 4 + 8 //object, hash, array ref
			+ 32 + value.length();     //char array
	}

	/**
	 *  This method performs the value comparison on two frames
	 *  if the values in both frames are equal, not equal, less than, greater than, less than/greater than and equal to
	 *  the output frame will store boolean value for each each comparison
	 *
	 *  @param bop binary operator
	 *  @param that frame block of rhs of m * n dimensions
	 *  @param out output frame block
	 *  @return a boolean frameBlock
	 */
	public FrameBlock binaryOperations(BinaryOperator bop, FrameBlock that, FrameBlock out) {
		if(getNumColumns() != that.getNumColumns() && getNumRows() != that.getNumColumns())
			throw new DMLRuntimeException("Frame dimension mismatch "+getNumRows()+" * "+getNumColumns()+
				" != "+that.getNumRows()+" * "+that.getNumColumns());
		String[][] outputData = new String[getNumRows()][getNumColumns()];
		//compare output value, incl implicit type promotion if necessary
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
			else if(getSchema()[i] == ValueType.FP64 || that
				.getSchema()[i] == ValueType.FP64 || getSchema()[i] == ValueType.FP32 || that
				.getSchema()[i] == ValueType.FP32) {
				for(int j = 0; j < getNumRows(); j++) {
					if(checkAndSetEmpty(frameBlock, that, outputData, j, i))
						continue;
					ScalarObject so1 = new DoubleObject(Double.parseDouble(get(j, i).toString()));
					ScalarObject so2 = new DoubleObject(Double.parseDouble(that.get(j, i).toString()));
					outputData[j][i] = String.valueOf(vcomp.compare(so1.getDoubleValue(), so2.getDoubleValue()));
				}
			}
			else if(getSchema()[i] == ValueType.INT64 || that
				.getSchema()[i] == ValueType.INT64 || getSchema()[i] == ValueType.INT32 || that
				.getSchema()[i] == ValueType.INT32) {
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

	///////
	// indexing and append operations

	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, IndexRange ixrange, FrameBlock ret) {
		return leftIndexingOperations(rhsFrame,
				(int)ixrange.rowStart, (int)ixrange.rowEnd,
				(int)ixrange.colStart, (int)ixrange.colEnd, ret);
	}

	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, int rl, int ru, int cl, int cu, FrameBlock ret) {
		// check the validity of bounds
		if (   rl < 0 || rl >= getNumRows() || ru < rl || ru >= getNumRows()
			|| cl < 0 || cu >= getNumColumns() || cu < cl || cu >= getNumColumns() ) {
			throw new DMLRuntimeException("Invalid values for frame indexing: ["+(rl+1)+":"+(ru+1)+"," + (cl+1)+":"+(cu+1)+"] " +
							"must be within frame dimensions ["+getNumRows()+","+getNumColumns()+"].");
		}

		if ( (ru-rl+1) < rhsFrame.getNumRows() || (cu-cl+1) < rhsFrame.getNumColumns()) {
			throw new DMLRuntimeException("Invalid values for frame indexing: " +
					"dimensions of the source frame ["+rhsFrame.getNumRows()+"x" + rhsFrame.getNumColumns() + "] " +
					"do not match the shape of the frame specified by indices [" +
					(rl+1) +":" + (ru+1) + ", " + (cl+1) + ":" + (cu+1) + "].");
		}


		//allocate output frame (incl deep copy schema)
		if( ret == null )
			ret = new FrameBlock();
		ret._numRows = _numRows;
		ret._schema = _schema.clone();
		ret._colnames = (_colnames != null) ? _colnames.clone() : null;
		ret._colmeta = _colmeta.clone();
		ret._coldata = new Array[getNumColumns()];

		//copy data to output and partial overwrite w/ rhs
		for( int j=0; j<getNumColumns(); j++ ) {
			Array tmp = _coldata[j].clone();
			if( j>=cl && j<=cu ) {
				//fast-path for homogeneous column schemas
				if( _schema[j]==rhsFrame._schema[j-cl] )
					tmp.set(rl, ru, rhsFrame._coldata[j-cl]);
				//general-path for heterogeneous column schemas
				else {
					for( int i=rl; i<=ru; i++ )
						tmp.set(i, UtilFunctions.objectToObject(
							_schema[j], rhsFrame._coldata[j-cl].get(i-rl)));
				}
			}
			ret._coldata[j] = tmp;
		}

		return ret;
	}

	public FrameBlock slice(IndexRange ixrange, FrameBlock ret) {
		return slice(
				(int)ixrange.rowStart, (int)ixrange.rowEnd,
				(int)ixrange.colStart, (int)ixrange.colEnd, ret);
	}

	@Override
	public FrameBlock slice(int rl, int ru, int cl, int cu, CacheBlock retCache) {
		return slice(rl, ru, cl, cu, false, retCache);
	}
	
	/**
	 * Right indexing operations to slice a subframe out of this frame block.
	 * Note that the existing column value types are preserved.
	 *
	 * @param rl row lower index, inclusive, 0-based
	 * @param ru row upper index, inclusive, 0-based
	 * @param cl column lower index, inclusive, 0-based
	 * @param cu column upper index, inclusive, 0-based
	 * @param deep enforce deep-copy
	 * @param retCache cache block
	 * @return frame block
	 */
	@Override
	public FrameBlock slice(int rl, int ru, int cl, int cu, boolean deep, CacheBlock retCache) {
		FrameBlock ret = (FrameBlock)retCache;
		// check the validity of bounds
		if (   rl < 0 || rl >= getNumRows() || ru < rl || ru >= getNumRows()
			|| cl < 0 || cu >= getNumColumns() || cu < cl || cu >= getNumColumns() ) {
			throw new DMLRuntimeException("Invalid values for frame indexing: ["+(rl+1)+":"+(ru+1)+"," + (cl+1)+":"+(cu+1)+"] " +
							"must be within frame dimensions ["+getNumRows()+","+getNumColumns()+"]");
		}

		//allocate output frame
		if( ret == null )
			ret = new FrameBlock();
		else
			ret.reset(ru-rl+1, true);

		//copy output schema and colnames
		int numCols = cu-cl+1;
		boolean isDefNames = isColNamesDefault();
		ret._schema = new ValueType[numCols];
		ret._colnames = !isDefNames ? new String[numCols] : null;
		ret._colmeta = new ColumnMetadata[numCols];

		for( int j=cl; j<=cu; j++ ) {
			ret._schema[j-cl] = _schema[j];
			ret._colmeta[j-cl] = _colmeta[j];
			if( !isDefNames )
				ret._colnames[j-cl] = getColumnName(j);
		}
		ret._numRows = ru-rl+1;
		if(ret._coldata == null )
			ret._coldata = new Array[numCols];

		//fast-path: shallow copy column indexing
		if( ret._numRows == _numRows && !deep ) {
			//this shallow copy does not only avoid an array copy, but
			//also allows for bi-directional reuses of recodemaps
			for( int j=cl; j<=cu; j++ )
				ret._coldata[j-cl] = _coldata[j];
		}
		//copy output data
		else {
			for( int j=cl; j<=cu; j++ ) {
				if( ret._coldata[j-cl] == null )
					ret._coldata[j-cl] = _coldata[j].slice(rl,ru);
				else
					ret._coldata[j-cl].set(0, ru-rl, _coldata[j], rl);
			}
		}
		return ret;
	}


	public void slice(ArrayList<Pair<Long,FrameBlock>> outlist, IndexRange range, int rowCut)
	{
		FrameBlock top=null, bottom=null;
		Iterator<Pair<Long,FrameBlock>> p=outlist.iterator();

		if(range.rowStart<rowCut)
			top = p.next().getValue();

		if(range.rowEnd>=rowCut)
			bottom = p.next().getValue();

		if(getNumRows() > 0)
		{
			int r=(int) range.rowStart;

			for(; r<Math.min(rowCut, range.rowEnd+1); r++)
			{
				Object[] row = new Object[(int) (range.colEnd-range.colStart+1)];
				for(int c=(int) range.colStart; c<range.colEnd+1; c++)
					row[(int) (c-range.colStart)] = get(r,c);
				top.appendRow(row);
			}

			for(; r<=range.rowEnd; r++)
			{
				Object[] row = new Object[(int) (range.colEnd-range.colStart+1)];
				for(int c=(int) range.colStart; c<range.colEnd+1; c++)
					row[(int) (c-range.colStart)] = get(r,c);
				bottom.appendRow(row);
			}
		}
	}

	/**
	 * Appends the given argument frameblock 'that' to this frameblock by
	 * creating a deep copy to prevent side effects. For cbind, the frames
	 * are appended column-wise (same number of rows), while for rbind the
	 * frames are appended row-wise (same number of columns).
	 *
	 * @param that frame block to append to current frame block
	 * @param ret frame block to return, can be null
	 * @param cbind if true, column append
	 * @return frame block
	 */
	public FrameBlock append( FrameBlock that, FrameBlock ret, boolean cbind ) {
		if( cbind ) //COLUMN APPEND
		{
			//sanity check row dimension mismatch
			if( getNumRows() != that.getNumRows() ) {
				throw new DMLRuntimeException("Incompatible number of rows for cbind: "+
						that.getNumRows()+" (expected: "+getNumRows()+")");
			}

			//allocate output frame
			if( ret == null )
				ret = new FrameBlock();
			ret._numRows = _numRows;

			//concatenate schemas (w/ deep copy to prevent side effects)
			ret._schema = (ValueType[]) ArrayUtils.addAll(_schema, that._schema);
			ret._colnames = (String[]) ArrayUtils.addAll(getColumnNames(), that.getColumnNames());
			ret._colmeta = (ColumnMetadata[]) ArrayUtils.addAll(_colmeta, that._colmeta);

			//check and enforce unique columns names
			if( !Arrays.stream(ret._colnames).allMatch(new HashSet<>()::add) )
				ret._colnames = createColNames(ret.getNumColumns());

			//concatenate column data (w/ shallow copy which is safe due to copy on write semantics)
			ret._coldata = (Array[]) ArrayUtils.addAll(_coldata, that._coldata);
		}
		else //ROW APPEND
		{
			//sanity check column dimension mismatch
			if( getNumColumns() != that.getNumColumns() ) {
				throw new DMLRuntimeException("Incompatible number of columns for rbind: "+
						that.getNumColumns()+" (expected: "+getNumColumns()+")");
			}

			//allocate output frame (incl deep copy schema)
			if( ret == null )
				ret = new FrameBlock();
			ret._numRows = _numRows;
			ret._schema = _schema.clone();
			ret._colnames = (_colnames!=null) ? _colnames.clone() : null;
			ret._colmeta = new ColumnMetadata[getNumColumns()];
			for( int j=0; j<_schema.length; j++ )
				ret._colmeta[j] = new ColumnMetadata(0);

			//concatenate data (deep copy first, append second)
			ret._coldata = new Array[getNumColumns()];
			for( int j=0; j<getNumColumns(); j++ )
				ret._coldata[j] = _coldata[j].clone();
			Iterator<Object[]> iter = that.getObjectRowIterator(_schema);
			while( iter.hasNext() )
				ret.appendRow(iter.next());
		}
		return ret;
	}

	public void copy(FrameBlock src) {
		copy(0, src.getNumRows()-1, 0, src.getNumColumns()-1, src);
	}

	public void copy(int rl, int ru, int cl, int cu, FrameBlock src)
	{
		//allocate columns if necessary
		ensureAllocatedColumns(ru-rl+1);

		//copy values
		for( int j=cl; j<=cu; j++ ) {
			//special case: column memcopy
			if( _schema[j].equals(src._schema[j-cl]) )
				_coldata[j].set(rl, ru, src._coldata[j-cl]);
			//general case w/ schema transformation
			else
				for( int i=rl; i<=ru; i++ ) {
					String tmp = src.get(i-rl, j-cl)!=null ? src.get(i-rl, j-cl).toString() : null;
					set(i, j, UtilFunctions.stringToObject(_schema[j], tmp));
				}
		}
	}


	///////
	// transform specific functionality

	/**
	 * This function will split every Recode map in the column using delimiter Lop.DATATYPE_PREFIX,
	 * as Recode map generated earlier in the form of Code+Lop.DATATYPE_PREFIX+Token and store it in a map
	 * which contains token and code for every unique tokens.
	 *
	 * @param col	is the column # from frame data which contains Recode map generated earlier.
	 * @return map of token and code for every element in the input column of a frame containing Recode map
	 */
	public HashMap<String,Long> getRecodeMap(int col) {
		//probe cache for existing map
		if( REUSE_RECODE_MAPS ) {
			SoftReference<HashMap<String,Long>> tmp = _coldata[col]._rcdMapCache;
			HashMap<String,Long> map = (tmp!=null) ? tmp.get() : null;
			if( map != null ) return map;
		}

		//construct recode map
		HashMap<String,Long> map = new HashMap<>();
		Array ldata = _coldata[col];
		for( int i=0; i<getNumRows(); i++ ) {
			Object val = ldata.get(i);
			if( val != null ) {
				String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(val.toString());
				map.put(tmp[0], Long.parseLong(tmp[1]));
			}
		}

		//put created map into cache
		if( REUSE_RECODE_MAPS )
			_coldata[col]._rcdMapCache = new SoftReference<>(map);

		return map;
	}

	@Override
	public void merge(CacheBlock that, boolean bDummy) {
		merge((FrameBlock)that);
	}

	public void merge(FrameBlock that) {
		//check for empty input source (nothing to merge)
		if( that == null || that.getNumRows() == 0 )
			return;

		//check dimensions (before potentially copy to prevent implicit dimension change)
		if ( getNumRows() != that.getNumRows() || getNumColumns() != that.getNumColumns() )
			throw new DMLRuntimeException("Dimension mismatch on merge disjoint (target="+getNumRows()+"x"+getNumColumns()+", source="+that.getNumRows()+"x"+that.getNumColumns()+")");

		//meta data copy if necessary
		for( int j=0; j<getNumColumns(); j++ )
			if( !that.isColumnMetadataDefault(j) ) {
				_colmeta[j].setNumDistinct(that._colmeta[j].getNumDistinct());
				_colmeta[j].setMvValue(that._colmeta[j].getMvValue());
			}

		//core frame block merge through cell copy
		//with column-wide access pattern
		for( int j=0; j<getNumColumns(); j++ ) {
			//special case: copy non-zeros of column
			if( _schema[j].equals(that._schema[j]) )
				_coldata[j].setNz(0, _numRows-1, that._coldata[j]);
			//general case w/ schema transformation
			else {
				for( int i=0; i<_numRows; i++ ) {
					Object obj = UtilFunctions.objectToObject(
							_schema[j], that.get(i,j), true);
					if (obj != null) //merge non-zeros
						set(i, j,obj);
				}
			}
		}
	}

	/**
	 * This function ZERO OUT the data in the slicing window applicable for this block.
	 *
	 * @param result frame block
	 * @param range index range
	 * @param complementary ?
	 * @param iRowStartSrc ?
	 * @param iRowStartDest ?
	 * @param blen ?
	 * @param iMaxRowsToCopy ?
	 * @return frame block
	 */
	public FrameBlock zeroOutOperations(FrameBlock result, IndexRange range, boolean complementary, int iRowStartSrc, int iRowStartDest, int blen, int iMaxRowsToCopy) {
		int clen = getNumColumns();

		if(result==null)
			result=new FrameBlock(getSchema());
		else
		{
			result.reset(0, true);
			result.setSchema(getSchema());
		}
		result.ensureAllocatedColumns(blen);

		if(complementary)
		{
			for(int r=(int) range.rowStart; r<=range.rowEnd&&r+iRowStartDest<blen; r++)
			{
				for(int c=(int) range.colStart; c<=range.colEnd; c++)
					result.set(r+iRowStartDest, c, get(r+iRowStartSrc,c));
			}
		}else
		{
			int r=iRowStartDest;
			for(; r<(int)range.rowStart && r-iRowStartDest<iMaxRowsToCopy ; r++)
				for(int c=0; c<clen; c++/*, offset++*/)
					result.set(r, c, get(r+iRowStartSrc-iRowStartDest,c));

			for(; r<=(int)range.rowEnd && r-iRowStartDest<iMaxRowsToCopy ; r++)
			{
				for(int c=0; c<(int)range.colStart; c++)
					result.set(r, c, get(r+iRowStartSrc-iRowStartDest,c));

				for(int c=(int)range.colEnd+1; c<clen; c++)
					result.set(r, c, get(r+iRowStartSrc-iRowStartDest,c));
			}

			for(; r-iRowStartDest<iMaxRowsToCopy ; r++)
				for(int c=0; c<clen; c++)
					result.set(r, c, get(r+iRowStartSrc-iRowStartDest,c));
		}

		return result;
	}

	public FrameBlock getSchemaTypeOf() {
		FrameBlock fb = new FrameBlock(
			UtilFunctions.nCopies(getNumColumns(), ValueType.STRING));
		fb.appendRow(Arrays.stream(_schema)
			.map(vt -> vt.toString()).toArray(String[]::new));
		return fb;
	}

	///////
	// row iterators (over strings and boxed objects)

	private abstract class RowIterator<T> implements Iterator<T[]> {
		protected final int[] _cols;
		protected final T[] _curRow;
		protected final int _maxPos;
		protected int _curPos = -1;

		protected RowIterator(int rl, int ru) {
			this(rl, ru, UtilFunctions.getSeqArray(1, getNumColumns(), 1));
		}

		protected RowIterator(int rl, int ru, int[] cols) {
			_curRow = createRow(cols.length);
			_cols = cols;
			_maxPos = ru;
			_curPos = rl;
		}

		@Override
		public boolean hasNext() {
			return (_curPos < _maxPos);
		}

		@Override
		public void remove() {
			throw new RuntimeException("RowIterator.remove is unsupported!");
		}

		protected abstract T[] createRow(int size);
	}

	private class StringRowIterator extends RowIterator<String> {
		public StringRowIterator(int rl, int ru) {
			super(rl, ru);
		}

		public StringRowIterator(int rl, int ru, int[] cols) {
			super(rl, ru, cols);
		}

		@Override
		protected String[] createRow(int size) {
			return new String[size];
		}

		@Override
		public String[] next( ) {
			for( int j=0; j<_cols.length; j++ ) {
				Object tmp = get(_curPos, _cols[j]-1);
				_curRow[j] = (tmp!=null) ? tmp.toString() : null;
			}
			_curPos++;
			return _curRow;
		}
	}

	private class ObjectRowIterator extends RowIterator<Object> {
		private ValueType[] _tgtSchema = null;

		public ObjectRowIterator(int rl, int ru) {
			super(rl, ru);
		}

		public ObjectRowIterator(int rl, int ru, int[] cols) {
			super(rl, ru, cols);
		}

		public void setSchema(ValueType[] schema) {
			_tgtSchema = schema;
		}

		@Override
		protected Object[] createRow(int size) {
			return new Object[size];
		}

		@Override
		public Object[] next( ) {
			for( int j=0; j<_cols.length; j++ )
				_curRow[j] = getValue(_curPos, _cols[j]-1);
			_curPos++;
			return _curRow;
		}

		private Object getValue(int i, int j) {
			Object val = get(i, j);
			if( _tgtSchema != null )
				val = UtilFunctions.objectToObject(_tgtSchema[j], val);
			return val;
		}
	}

	///////
	// generic, resizable native arrays

	/**
	 * Base class for generic, resizable array of various value types. We
	 * use this custom class hierarchy instead of Trove or other libraries
	 * in order to avoid unnecessary dependencies.
	 */
	private abstract static class Array<T> implements Writable {
		protected SoftReference<HashMap<String,Long>> _rcdMapCache = null;

		protected int _size = 0;
		protected int newSize() {
			return Math.max(_size*2, 4);
		}
		public abstract T get(int index);
		public abstract void set(int index, T value);
		public abstract void set(int rl, int ru, Array value);
		public abstract void set(int rl, int ru, Array value, int rlSrc);
		public abstract void setNz(int rl, int ru, Array value);
		public abstract void append(String value);
		public abstract void append(T value);
		@Override
		public abstract Array clone();
		public abstract Array slice(int rl, int ru);
		public abstract void reset(int size);

		@Override
		public String toString(){
			return this.getClass().getSimpleName().toString() + ":" + _size;
		}
	}

	private static class StringArray extends Array<String> {
		private String[] _data = null;

		public StringArray(String[] data) {
			_data = data;
			_size = _data.length;
		}
		public String[] get() { return _data; }

		@Override
		public String get(int index) {
			return _data[index];
		}

		@Override
		public void set(int index, String value) {
			_data[index] = value;
		}
		@Override
		public void set(int rl, int ru, Array value) {
			set(rl, ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((StringArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			String[] data2 = ((StringArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i]!=null )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeUTF((_data[i]!=null)?_data[i]:"");
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ ) {
				String tmp = in.readUTF();
				_data[i] = (!tmp.isEmpty()) ? tmp : null;
			}
		}
		@Override
		public Array clone() {
			return new StringArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new StringArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new String[size];
			_size = size;
		}
	}

	private static class BooleanArray extends Array<Boolean> {
		private boolean[] _data = null;

		public BooleanArray(boolean[] data) {
			_data = data;
			_size = _data.length;
		}
		public boolean[] get() { return _data; }

		@Override
		public Boolean get(int index) {
			return _data[index];
		}

		@Override
		public void set(int index, Boolean value) {
			_data[index] = (value!=null) ? value : false;
		}
		@Override
		public void set(int rl, int ru, Array value) {
			set(rl, ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((BooleanArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			boolean[] data2 = ((BooleanArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i] )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {
			append(Boolean.parseBoolean(value));
		}
		@Override
		public void append(Boolean value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : false;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeBoolean(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readBoolean();
		}
		@Override
		public Array clone() {
			return new BooleanArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new BooleanArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new boolean[size];
			_size = size;
		}
	}

	private static class LongArray extends Array<Long> {
		private long[] _data = null;

		public LongArray(long[] data) {
			_data = data;
			_size = _data.length;
		}
		public long[] get() { return _data; }
		@Override
		public Long get(int index) {
			return _data[index];
		}
		@Override
		public void set(int index, Long value) {
			_data[index] = (value!=null) ? value : 0L;
		}
		@Override
		public void set(int rl, int ru, Array value) {
			set(rl, ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((LongArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			long[] data2 = ((LongArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i]!=0 )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {
			append((value!=null)?Long.parseLong(value.trim()):null);
		}
		@Override
		public void append(Long value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0L;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeLong(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readLong();
		}
		@Override
		public Array clone() {
			return new LongArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new LongArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new long[size];
			_size = size;
		}
	}

	private static class IntegerArray extends Array<Integer> {
		private int[] _data = null;

		public IntegerArray(int[] data) {
			_data = data;
			_size = _data.length;
		}
		public int[] get() { return _data; }

		@Override
		public Integer get(int index) {
			return _data[index];
		}

		@Override
		public void set(int index, Integer value) { _data[index] = (value!=null) ? value : 0;}
		@Override
		public void set(int rl, int ru, Array value) {
			set(rl, ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((IntegerArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			int[] data2 = ((IntegerArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i]!=0 )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {
			append((value!=null)?Integer.parseInt(value.trim()):null);
		}
		@Override
		public void append(Integer value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeLong(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readInt();
		}
		@Override
		public Array clone() {
			return new IntegerArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new IntegerArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new int[size];
			_size = size;
		}
	}

	private static class FloatArray extends Array<Float> {
		private float[] _data = null;

		public FloatArray(float[] data) {
			_data = data;
			_size = _data.length;
		}
		public float[] get() { return _data; }

		@Override
		public Float get(int index) {
			return _data[index];
		}

		@Override
		public void set(int index, Float value) { _data[index] = (value!=null) ? value : 0f; }

		@Override
		public void set(int rl, int ru, Array value) {
			set(rl,ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((FloatArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			float[] data2 = ((FloatArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i]!=0 )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {	append((value!=null)? Float.parseFloat(value):null); }

		@Override
		public void append(Float value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0f;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeFloat(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readFloat();
		}
		@Override
		public Array clone() {
			return new FloatArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new FloatArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new float[size];
			_size = size;
		}
	}

	private static class DoubleArray extends Array<Double> {
		private double[] _data = null;

		public DoubleArray(double[] data) {
			_data = data;
			_size = _data.length;
		}
		public double[] get() { return _data; }
		@Override
		public Double get(int index) {
			return _data[index];
		}
		@Override
		public void set(int index, Double value) {
			_data[index] = (value!=null) ? value : 0d;
		}
		@Override
		public void set(int rl, int ru, Array value) {
			set(rl,ru, value, 0);
		}
		@Override
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((DoubleArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		@Override
		public void setNz(int rl, int ru, Array value) {
			double[] data2 = ((DoubleArray)value)._data;
			for( int i=rl; i<ru+1; i++ )
				if( data2[i]!=0 )
					_data[i] = data2[i];
		}
		@Override
		public void append(String value) {
			append((value!=null)?Double.parseDouble(value):null);
		}
		@Override
		public void append(Double value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0d;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeDouble(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readDouble();
		}
		@Override
		public Array clone() {
			return new DoubleArray(Arrays.copyOf(_data, _size));
		}
		@Override
		public Array slice(int rl, int ru) {
			return new DoubleArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
		@Override
		public void reset(int size) {
			if( _data.length < size )
				_data = new double[size];
			_size = size;
		}
	}

	public static class ColumnMetadata implements Serializable {
		private static final long serialVersionUID = -90094082422100311L;

		private long _ndistinct = 0;
		private String _mvValue = null;

		public ColumnMetadata(long ndistinct) {
			_ndistinct = ndistinct;
		}
		public ColumnMetadata(long ndistinct, String mvval) {
			_ndistinct = ndistinct;
			_mvValue = mvval;
		}
		public ColumnMetadata(ColumnMetadata that) {
			_ndistinct = that._ndistinct;
			_mvValue = that._mvValue;
		}

		public long getNumDistinct() {
			return _ndistinct;
		}
		public void setNumDistinct(long ndistinct) {
			_ndistinct = ndistinct;
		}
		public String getMvValue() {
			return _mvValue;
		}
		public void setMvValue(String mvVal) {
			_mvValue = mvVal;
		}
	}

	private static ValueType isType(String val) {
		val = val.trim().toLowerCase().replaceAll("\"",  "");
		if (val.matches("(true|false|t|f|0|1)"))
			return ValueType.BOOLEAN;
		else if (val.matches("[-+]?\\d+")){
			long maxValue = Long.parseLong(val);
			if ((maxValue >= Integer.MIN_VALUE) && (maxValue <= Integer.MAX_VALUE))
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if (val.matches("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?")){
			double maxValue = Double.parseDouble(val);
			if ((maxValue >= (-Float.MAX_VALUE)) && (maxValue <= Float.MAX_VALUE))
				return ValueType.FP32;
			else
				return ValueType.FP64;
		}
		else if (val.equals("infinity") || val.equals("-infinity") || val.equals("nan"))
			return ValueType.FP64;
		else return ValueType.STRING;
	}

	public FrameBlock detectSchemaFromRow(double sampleFraction) {
		int rows = this.getNumRows();
		int cols = this.getNumColumns();
		String[] schemaInfo = new String[cols];
		int sample = (int)Math.min(Math.max(sampleFraction*rows, 256), rows);

		ExecutorService pool = CommonThreadPool.get(cols);
		ArrayList<DetectValueTypeTask> tasks = new ArrayList<>();
		for (int i = 0; i < cols; i++) {
			FrameBlock.Array obj = this.getColumn(i);
			tasks.add(new DetectValueTypeTask(obj,rows, sample));
		}

		List<Future<String>> ret;

		try {
			ret = pool.invokeAll(tasks);
			pool.shutdown();
			for(int i = 0; i < cols; i++){
				schemaInfo[i] = ret.get(i).get();
			}
		} catch (ExecutionException | InterruptedException e) {
			throw new DMLRuntimeException("Exception Interupted or Exception thrown in Detect Schema", e);
		}

		//create output block one row representing the schema as strings
		FrameBlock fb = new FrameBlock(UtilFunctions.nCopies(cols, ValueType.STRING));
		fb.appendRow(schemaInfo);
		return fb;
	}

	private static class DetectValueTypeTask implements Callable<String>
	{
		private final Array _obj;
		private final int _rows;
		private final int _sampleSize;


		protected DetectValueTypeTask(Array obj, int rows, int sampleSize ) {
			_obj = obj;
			_rows = rows;
			_sampleSize = sampleSize;
		}

		@Override
		public String call() {
			ValueType state = ValueType.UNKNOWN;
			for (int j = 0; j < _sampleSize; j++) {
				int randomIndex = ThreadLocalRandom.current().nextInt(0, _rows - 1);
				String dataValue = ((_obj.get(randomIndex) != null)?_obj.get(randomIndex).toString().trim().replace("\"", "").toLowerCase():null);
				if(dataValue != null){
					ValueType current = isType(dataValue);
					if (current == ValueType.STRING) {
						state = ValueType.STRING;
						break;
					}
					else if (current== ValueType.FP64) {
						state = ValueType.FP64;
					}
					else if (current== ValueType.FP32) {
						state = (state == ValueType.FP64 ? state : ValueType.FP32);
					}
					else if (current == ValueType.INT64) {
						state = ((state == ValueType.FP64  || state == ValueType.FP32) ? state : ValueType.INT64);
					}
					else if (current == ValueType.INT32) {
						state = ((state == ValueType.FP64 || state == ValueType.FP32 || state == ValueType.INT64) ? state : ValueType.INT32);
					}
					else if (current == ValueType.BOOLEAN)
						state = ((state == ValueType.FP64 || state == ValueType.FP32 || state == ValueType.INT64 || state == ValueType.INT32) ? state : ValueType.BOOLEAN);
				}
			}
			return state.name();
		}
	}

	/**
	 * Drop the cell value which does not confirms to the data type of its column
	 * @param schema of the frame
	 * @return original frame where invalid values are replaced with null
	 */
	public FrameBlock dropInvalidType(FrameBlock schema) {
		//sanity checks
		if(this.getNumColumns() != schema.getNumColumns())
			throw new DMLException("mismatch in number of columns in frame and its schema "+this.getNumColumns()+" != "+schema.getNumColumns());

		String[] schemaString = schema.getStringRowIterator().next(); // extract the schema in String array
		for (int i = 0; i < this.getNumColumns(); i++) {
			Array obj = this.getColumn(i);
			String schemaCol = schemaString[i];
			String type;
			if(schemaCol.contains("FP")){
				type = "FP";
			} else if (schemaCol.contains("INT")){
				type = "INT";
			} else if (schemaCol.contains("STRING")){
				// In case of String columns, don't do any verification or replacements.
				continue;
			} else{
				type = schemaCol;
			}

			for (int j = 0; j < this.getNumRows(); j++)
			{
				if(obj.get(j) == null)
					continue;
				String dataValue = obj.get(j).toString().trim().replace("\"", "").toLowerCase() ;

				ValueType dataType = isType(dataValue);

				if(!dataType.toString().contains(type) && !(dataType == ValueType.BOOLEAN && type.equals("INT")) &&
					!(dataType == ValueType.BOOLEAN && type.equals("FP"))){
					LOG.warn("Datatype detected: " + dataType + " where expected: " + schemaString[i] + " col: " +
						(i+1) + ", row:" +(j+1));

					this.set(j,i,null);
				}
			}
		}
		return this;
	}

	/**
	 *  This method validates the frame data against an attribute length constrain
	 *  if data value in any cell is greater than the specified threshold of that attribute
	 *  the output frame will store a null on that cell position, thus removing the length-violating values.
	 *
	 *  @param feaLen vector of valid lengths
	 *  @return FrameBlock with invalid values converted into missing values (null)
	 */
	public FrameBlock invalidByLength(MatrixBlock feaLen) {
		//sanity checks
		if(this.getNumColumns() != feaLen.getNumColumns())
			throw new DMLException("mismatch in number of columns in frame and corresponding feature-length vector");

		FrameBlock outBlock = new FrameBlock(this);
		for (int i = 0; i < this.getNumColumns(); i++) {
			if(feaLen.quickGetValue(0, i) == -1)
				continue;
			int validLength = (int)feaLen.quickGetValue(0, i);
			Array obj = this.getColumn(i);
			for (int j = 0; j < obj._size; j++)
			{
				if(obj.get(j) == null)
					continue;
				String dataValue = obj.get(j).toString();
				if(dataValue.length() > validLength)
					outBlock.set(j, i, null);
			}
		}

		return outBlock;
	}

	public static FrameBlock mergeSchema(FrameBlock temp1, FrameBlock temp2) {
		String[] rowTemp1 = temp1.getStringRowIterator().next();
		String[] rowTemp2 = temp2.getStringRowIterator().next();

		if(rowTemp1.length != rowTemp2.length)
			throw new DMLRuntimeException("Schema dimension "
				+ "mismatch: "+rowTemp1.length+" vs "+rowTemp2.length);

		for(int i=0; i< rowTemp1.length; i++ ) {
			//modify schema1 if necessary (different schema2)
			if(!rowTemp1[i].equals(rowTemp2[i])) {
				if(rowTemp1[i].equals("STRING") || rowTemp2[i].equals("STRING"))
					rowTemp1[i] = "STRING";
				else if (rowTemp1[i].equals("FP64") || rowTemp2[i].equals("FP64"))
					rowTemp1[i] = "FP64";
				else if (rowTemp1[i].equals("FP32") && new ArrayList<>(Arrays.asList("INT64", "INT32", "CHARACTER")).contains(rowTemp2[i]) )
					rowTemp1[i] = "FP32";
				else if (rowTemp1[i].equals("INT64") && new ArrayList<>(Arrays.asList("INT32", "CHARACTER")).contains(rowTemp2[i]))
					rowTemp1[i] = "INT64";
				else if (rowTemp1[i].equals("INT32") || rowTemp2[i].equals("CHARACTER"))
					rowTemp1[i] = "INT32";
			}
		}

		//create output block one row representing the schema as strings
		FrameBlock mergedFrame = new FrameBlock(
			UtilFunctions.nCopies(temp1.getNumColumns(), ValueType.STRING));
		mergedFrame.appendRow(rowTemp1);
		return mergedFrame;
	}
	
	public void mapInplace(Function<String, String> fun) {
		for(int j=0; j<getNumColumns(); j++)
			for(int i=0; i<getNumRows(); i++) {
				Object tmp = get(i, j);
				set(i, j, (tmp == null) ? tmp :
					UtilFunctions.objectToObject(_schema[j], fun.apply(tmp.toString())));
			}
	}

	public FrameBlock map (String lambdaExpr, long margin){
		if(!lambdaExpr.contains("->")) {
			String args = lambdaExpr.substring(lambdaExpr.indexOf('(') + 1, lambdaExpr.indexOf(')'));
			if(args.contains(",")) {
				String[] arguments = args.split(",");
				return DMVUtils.syntacticalPatternDiscovery(this, Double.parseDouble(arguments[0]), arguments[1]);
			} else if (args.contains(";")) {
				String[] arguments = args.split(";");
				return EMAUtils.exponentialMovingAverageImputation(this, Integer.parseInt(arguments[0]), arguments[1],
					Integer.parseInt(arguments[2]), Double.parseDouble(arguments[3]), Double.parseDouble(arguments[4]), Double.parseDouble(arguments[5]));
			}
		}
		if(lambdaExpr.contains("jaccardSim"))
			return mapDist(getCompiledFunction(lambdaExpr, margin));
		return map(getCompiledFunction(lambdaExpr, margin), margin);
	}

	public FrameBlock valueSwap(FrameBlock schema) {
		String[] schemaString = schema.getStringRowIterator().next();
		String dataValue2 = null;
		double minSimScore = 0;
		int bestIdx = 0;
		//		remove the precision info
		for(int i = 0; i < schemaString.length; i++)
			schemaString[i] = schemaString[i].replaceAll("\\d", "");

		double[] minColLength = new double[this.getNumColumns()];
		double[] maxColLength = new double[this.getNumColumns()];

		for(int k = 0; k < this.getNumColumns(); k++) {
			String[] data = ((StringArray) this.getColumn(k))._data;

			double minLength = Arrays.stream(data).filter(Objects::nonNull).mapToDouble(String::length).min().orElse(Double.NaN);
			double maxLength = Arrays.stream(data).filter(Objects::nonNull).mapToDouble(String::length).max().orElse(Double.NaN);
			maxColLength[k] = maxLength;
			minColLength[k] = minLength;
		}
		ArrayList<Integer> probColList = new ArrayList();
		for(int i = 0; i < this.getNumColumns(); i++) {
			for(int j = 0; j < this.getNumRows(); j++) {
				if(this.get(j, i) == null)
					continue;
				String dataValue = this.get(j, i).toString().trim().replace("\"", "").toLowerCase();
				ValueType dataType = isType(dataValue);

				String type = dataType.toString().replaceAll("\\d", "");
				//				get the avergae column length
				if(!dataType.toString().contains(schemaString[i]) && !(dataType == ValueType.BOOLEAN && schemaString[i]
					.equals("INT")) && !(dataType == ValueType.BOOLEAN && schemaString[i].equals("FP")) && !(dataType
					.toString().contains("INT") && schemaString[i].equals("FP"))) {
					LOG.warn("conflict " + dataType + " " + schemaString[i] + " " + dataValue);
					//					check the other column with satisfy the data type of this value
					for(int w = 0; w < schemaString.length; w++) {
						if(schemaString[w].equals(type) && dataValue.length() > minColLength[w] && dataValue
							.length() < maxColLength[w] && (w != i)) {
							Object item = this.get(j, w);
							String dataValueProb = (item != null) ? item.toString().trim().replace("\"", "")
								.toLowerCase() : "0";
							ValueType dataTypeProb = isType(dataValueProb);
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
							int randomIndex = ThreadLocalRandom.current().nextInt(0, _numRows - 1);
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

	public FrameBlock map (FrameMapFunction lambdaExpr, long margin) {
		// Prepare temporary output array
		String[][] output = new String[getNumRows()][getNumColumns()];

		if (margin == 1) {
			// Execute map function on rows
			for(int i = 0; i < getNumRows(); i++) {
				String[] row = new String[getNumColumns()];
				for(int j = 0; j < getNumColumns(); j++) {
					Array input = getColumn(j);
					row[j] = String.valueOf(input.get(i));
				}
				output[i] = lambdaExpr.apply(row);
			}
		} else if (margin == 2) {
			// Execute map function on columns
			for(int j = 0; j < getNumColumns(); j++) {
				String[] actualColumn = Arrays.copyOfRange((String[]) getColumnData(j), 0, getNumRows()); // since more rows can be allocated, mutable array
				String[] outColumn = lambdaExpr.apply(actualColumn);

				for(int i = 0; i < getNumRows(); i++)
					output[i][j] = outColumn[i];
			}
		} else {
			// Execute map function on all cells
			for(int j = 0; j < getNumColumns(); j++) {
				Array input = getColumn(j);
				for(int i = 0; i < input._size; i++)
					if(input.get(i) != null)
						output[i][j] = lambdaExpr.apply(String.valueOf(input.get(i)));
			}
		}
		return new FrameBlock(UtilFunctions.nCopies(getNumColumns(), ValueType.STRING), output);
	}

	public FrameBlock mapDist (FrameMapFunction lambdaExpr) {
		String[][] output = new String[getNumRows()][getNumRows()];
		for(String[] row : output)
			Arrays.fill(row, "0.0");
		Array input = getColumn(0);
		for(int j = 0; j < input._size - 1; j++) {
			for(int i = j + 1; i < input._size; i++)
				if(input.get(i) != null && input.get(j) != null) {
					output[j][i] = lambdaExpr.apply(String.valueOf(input.get(j)), String.valueOf(input.get(i)));
				}
		}
		return new FrameBlock(UtilFunctions.nCopies(getNumRows(), ValueType.STRING), output);
	}

	public static FrameMapFunction getCompiledFunction (String lambdaExpr, long margin) {
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
		sb.append("import org.apache.sysds.runtime.matrix.data.FrameBlock.FrameMapFunction;\n");
		sb.append("import java.util.Arrays;\n");
		sb.append("public class " + cname + " extends FrameMapFunction {\n");
		if(margin != 0) {
			sb.append("public String[] apply(String[] " + varname[0].trim() + ") {\n");
			sb.append("  return UtilFunctions.toStringArray(" + expr + "); }}\n");
		} else {
			if(varname.length == 1) {
				sb.append("public String apply(String " + varname[0].trim() + ") {\n");
				sb.append("  return String.valueOf(" + expr + "); }}\n");
			}
			else if(varname.length == 2) {
				sb.append("public String apply(String " + varname[0].trim() + ", String " + varname[1].trim() + ") {\n");
				sb.append("  return String.valueOf(" + expr + "); }}\n");
			}
		}
		// compile class, and create FrameMapFunction object
		try {
			return (FrameMapFunction) CodegenUtils.compileClass(cname, sb.toString())
					.getDeclaredConstructor().newInstance();
		}
		catch(InstantiationException | IllegalAccessException 
			| IllegalArgumentException | InvocationTargetException
			| NoSuchMethodException | SecurityException e) {
			throw new DMLRuntimeException("Failed to compile FrameMapFunction.", e);
		}
	}

	public static class FrameMapFunction implements Serializable {
		private static final long serialVersionUID = -8398572153616520873L;
		public String apply(String input) {return null;}
		public String apply(String input1, String input2) {	return null;}
		public String[] apply(String[] input1) {	return null;}
	}

	public <T> FrameBlock replaceOperations(String pattern, String replacement) {
		FrameBlock ret = new FrameBlock(this);

		ValueType patternType = UtilFunctions.isBoolean(pattern) ? ValueType.BOOLEAN : (NumberUtils.isCreatable(pattern) ?
			(UtilFunctions.isIntegerNumber(pattern) ? ValueType.INT64 : ValueType.FP64) : ValueType.STRING);
		ValueType replacementType = UtilFunctions.isBoolean(replacement) ? ValueType.BOOLEAN : (NumberUtils.isCreatable(replacement) ?
			(UtilFunctions.isIntegerNumber(replacement) ? ValueType.INT64 : ValueType.FP64) : ValueType.STRING);

		if(patternType != replacementType || !ValueType.isSameTypeString(patternType, replacementType))
			throw new DMLRuntimeException("Pattern and replacement types should be same.");

		for(int i = 0; i < ret.getNumColumns(); i++){
			Array colData = ret._coldata[i];
			for(int j = 0; j < colData._size && (ValueType.isSameTypeString(_schema[i], patternType) || _schema[i] == ValueType.STRING); j++) {
				T patternNew =  (T) UtilFunctions.stringToObject(_schema[i], pattern);
				T replacementNew = (T) UtilFunctions.stringToObject(_schema[i], replacement);

				Object ent = colData.get(j);
				if(ent != null && ent.toString().equals(patternNew.toString()))
					colData.set(j,replacementNew);
				else  if(ent instanceof String && ent.equals(pattern))
					colData.set(j, replacement);
			}
		}
		return ret;
	}

	public  FrameBlock removeEmptyOperations(boolean rows, boolean emptyReturn, MatrixBlock select) {
		if( rows )
			return removeEmptyRows(select, emptyReturn);
		else //cols
			return removeEmptyColumns(select, emptyReturn);
	}

	private FrameBlock removeEmptyRows(MatrixBlock select, boolean emptyReturn) {
		if(select != null && (select.getNumRows() != getNumRows() && select.getNumColumns() != getNumRows()))
			throw new DMLRuntimeException("Frame rmempty: Incorrect select vector dimensions.");

		FrameBlock ret = new FrameBlock(_schema, _colnames);

		if (select == null) {
			Object[] row = new Object[getNumColumns()];
			for(int i = 0; i < _numRows; i++) {
				boolean isEmpty = true;
				for(int j = 0; j < getNumColumns(); j++) {
					row[j] = _coldata[j].get(i);
					isEmpty &= ArrayUtils.contains(new double[]{0.0, Double.NaN},
						UtilFunctions.objectToDoubleSafe(_schema[j], row[j]));
				}
				if(!isEmpty)
					ret.appendRow(row);
			}
		}
		else {
			if(select.getNonZeros() == getNumRows())
				return this;

			int[] indices = DataConverter.convertVectorToIndexList(select);
			
			Object[] row = new Object[getNumColumns()];
			for(int i : indices) {
				for(int j = 0; j < getNumColumns(); j++)
					row[j] = _coldata[j].get(i);
				ret.appendRow(row);
			}
		}

		if (ret.getNumRows() == 0 && emptyReturn) {
			String[][] arr = new String[1][getNumColumns()];
			Arrays.fill(arr, new String[]{null});
			ValueType[] schema = new ValueType[getNumColumns()];
			Arrays.fill(schema, ValueType.STRING);
			return new FrameBlock(schema, arr);
		}

		return ret;
	}

	private FrameBlock removeEmptyColumns(MatrixBlock select, boolean emptyReturn) {
		if(select != null && (select.getNumRows() != getNumColumns() && select.getNumColumns() != getNumColumns())) {
			throw new DMLRuntimeException("Frame rmempty: Incorrect select vector dimensions.");
		}

		FrameBlock ret = new FrameBlock();
		List<ColumnMetadata> columnMetadata = new ArrayList<>();

		if (select == null) {
			for(int i = 0; i < getNumColumns(); i++) {
				Array colData = _coldata[i];
				ValueType type = _schema[i];
				boolean isEmpty = IntStream.range(0, colData._size)
					.mapToObj((IntFunction<Object>) colData::get)
					.allMatch(e -> ArrayUtils.contains(new double[]{0.0, Double.NaN}, UtilFunctions.objectToDoubleSafe(type, e)));

				if(!isEmpty) {
					ret.appendColumn(_schema[i], _coldata[i]);
					columnMetadata.add(new ColumnMetadata(_colmeta[i]));
				}
			}
		} else {
			if(select.getNonZeros() == getNumColumns())
				return new FrameBlock(this);

			int[] indices = DataConverter.convertVectorToIndexList(select);
			int k = 0;
			for(int i : indices) {
				ret.appendColumn(_schema[i], _coldata[i]);
				columnMetadata.add(new ColumnMetadata(_colmeta[i]));
				if(_colnames != null)
					ret._colnames[k++] = _colnames[i];
			}
		}

		if(ret.getNumColumns() == 0 && emptyReturn) {
			String[][] arr = new String[_numRows][];
			Arrays.fill(arr, new String[]{null});
			return new FrameBlock(new ValueType[]{ValueType.STRING}, arr);
		}

		ret._colmeta = new ColumnMetadata[columnMetadata.size()];
		columnMetadata.toArray(ret._colmeta);
		ret.setColumnMetadata(ret._colmeta);

		return ret;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("FrameBlock");
		sb.append("\n");
		sb.append(Arrays.toString(_schema));
		sb.append("\n");
		sb.append(Arrays.toString(_coldata));

		return sb.toString();
	}
}
