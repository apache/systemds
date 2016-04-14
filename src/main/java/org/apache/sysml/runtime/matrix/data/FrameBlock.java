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

package org.apache.sysml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.io.Writable;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * 
 */
@SuppressWarnings({"rawtypes","unchecked"}) //allow generic native arrays
public class FrameBlock implements Writable, CacheBlock, Externalizable
{
	private static final long serialVersionUID = -3993450030207130665L;
	
	//internal configuration
	private static final boolean REUSE_RECODE_MAPS = true;
	
	/** The number of rows of the FrameBlock */
	private int _numRows = -1;
	
	/** The schema of the data frame as an ordered list of value types */
	private List<ValueType> _schema = null; 
	
	/** The column names of the data frame as an ordered list of strings */
	private List<String> _colnames = null;
	
	/** The data frame data as an ordered list of columns */
	private List<Array> _coldata = null;
	
	/** Cache for recode maps from frame meta data, indexed by column 0-based */
	private Map<Integer, SoftReference<HashMap<String,Long>>> _rcdMapCache = null;
	
	public FrameBlock() {
		_numRows = 0;
		_schema = new ArrayList<ValueType>();
		_colnames = new ArrayList<String>();
		_coldata = new ArrayList<Array>();
		if( REUSE_RECODE_MAPS )
			_rcdMapCache = new HashMap<Integer, SoftReference<HashMap<String,Long>>>();
	}
	
	public FrameBlock(FrameBlock that) {
		this(that.getSchema());
		copy(that);
	}
	
	public FrameBlock(int ncols, ValueType vt) {
		this();
		_schema.addAll(Collections.nCopies(ncols, vt));
		_colnames = createColNames(ncols);
	}
	
	public FrameBlock(List<ValueType> schema) {
		this(schema, new String[0][]);
	}
	
	public FrameBlock(List<ValueType> schema, List<String> names) {
		this(schema, names, new String[0][]);
	}
	
	public FrameBlock(List<ValueType> schema, String[][] data) {
		this(schema, createColNames(schema.size()), data);
	}
	
	public FrameBlock(List<ValueType> schema, List<String> names, String[][] data) {
		_numRows = 0; //maintained on append
		_schema = new ArrayList<ValueType>(schema);
		_colnames = new ArrayList<String>(names);
		_coldata = new ArrayList<Array>();
		for( int i=0; i<data.length; i++ )
			appendRow(data[i]);
		if( REUSE_RECODE_MAPS )
			_rcdMapCache = new HashMap<Integer, SoftReference<HashMap<String,Long>>>();
	}
	
	/**
	 * Get the number of rows of the frame block.
	 * 
	 * @return 
	 */
	public int getNumRows() {
		return _numRows;
	}
	
	/**
	 * Get the number of columns of the frame block, that is
	 * the number of columns defined in the schema.
	 * 
	 * @return
	 */
	public int getNumColumns() {
		return _schema.size();
	}
	
	/**
	 * Returns the schema of the frame block.
	 * 
	 * @return
	 */
	public List<ValueType> getSchema() {
		return _schema;
	}
	
	/**
	 * Returns the column names of the frame block.
	 * 
	 * @return
	 */
	public List<String> getColumnNames() {
		return _colnames;
	}
	
	/**
	 * Creates a mapping from column names to column IDs, i.e., 
	 * 1-based column indexes
	 * 
	 * @return
	 */
	public Map<String,Integer> getColumnNameIDMap() {
		Map<String, Integer> ret = new HashMap<String, Integer>();
		for( int j=0; j<getNumColumns(); j++ )
			ret.put(_colnames.get(j), j+1);
		return ret;	
	}
	
	/**
	 * Allocate column data structures if necessary, i.e., if schema specified
	 * but not all column data structures created yet.
	 */
	public void ensureAllocatedColumns(int numRows) {
		//early abort if already allocated
		if( _schema.size() == _coldata.size() ) 
			return;		
		//allocate columns if necessary
		for( int j=0; j<_schema.size(); j++ ) {
			if( j >= _coldata.size() )
				switch( _schema.get(j) ) {
					case STRING:  _coldata.add(new StringArray(new String[numRows])); break;
					case BOOLEAN: _coldata.add(new BooleanArray(new boolean[numRows])); break;
					case INT:     _coldata.add(new LongArray(new long[numRows])); break;
					case DOUBLE:  _coldata.add(new DoubleArray(new double[numRows])); break;
					default: throw new RuntimeException("Unsupported value type: "+_schema.get(j));
				}
		}
		_numRows = numRows;
	}
	
	/**
	 * Checks for matching column sizes in case of existing columns.
	 * 		
	 * @param newlen
	 */
	public void ensureColumnCompatibility(int newlen) {
		if( _coldata.size() > 0 && _numRows != newlen )
			throw new RuntimeException("Mismatch in number of rows: "+newlen+" (expected: "+_numRows+")");
	}
	
	/**
	 * 
	 * @param size
	 * @return
	 */
	private static List<String> createColNames(int size) {
		ArrayList<String> ret = new ArrayList<String>(size);
		for( int i=1; i<=size; i++ )
			ret.add("C"+i);
		return ret;
	}
	
	///////
	// basic get and set functionality
	
	/**
	 * Gets a boxed object of the value in position (r,c).
	 * 
	 * @param r	row index, 0-based
	 * @param c	column index, 0-based
	 * @return
	 */
	public Object get(int r, int c) {
		return _coldata.get(c).get(r);
	}
	
	/**
	 * Sets the value in position (r,c), where the input is assumed
	 * to be a boxed object consistent with the schema definition.
	 * 
	 * @param r
	 * @param c
	 * @param val
	 */
	public void set(int r, int c, Object val) {
		_coldata.get(c).set(r, val);
	}
	
	public void reset(int nrow) 
	{
		getSchema().clear();
		getColumnNames().clear();
		if(_coldata != null)
			for(int i=0; i < _coldata.size(); ++i)
				_coldata.get(i)._size = nrow;
	}

	public void reset() 
	{
		reset(0);
	}
	

	/**
	 * Append a row to the end of the data frame, where all row fields
	 * are boxed objects according to the schema.
	 * 
	 * @param row
	 */
	public void appendRow(Object[] row) {
		ensureAllocatedColumns(0);
		for( int j=0; j<row.length; j++ )
			_coldata.get(j).append(row[j]);
		_numRows++;
	}
	
	/**
	 * Append a row to the end of the data frame, where all row fields
	 * are string encoded.
	 * 
	 * @param row
	 */
	public void appendRow(String[] row) {
		ensureAllocatedColumns(0);
		for( int j=0; j<row.length; j++ )
			_coldata.get(j).append(row[j]);
		_numRows++;
	}
	
	/**
	 * Append a column of value type STRING as the last column of 
	 * the data frame. The given array is wrapped but not copied 
	 * and hence might be updated in the future.
	 * 
	 * @param col
	 */
	public void appendColumn(String[] col) {
		ensureColumnCompatibility(col.length);
		_schema.add(ValueType.STRING);
		_coldata.add(new StringArray(col));
		_numRows = col.length;
	}
	
	/**
	 * Append a column of value type BOOLEAN as the last column of 
	 * the data frame. The given array is wrapped but not copied 
	 * and hence might be updated in the future.
	 * 
	 * @param col
	 */
	public void appendColumn(boolean[] col) {
		ensureColumnCompatibility(col.length);
		_schema.add(ValueType.BOOLEAN);
		_coldata.add(new BooleanArray(col));	
		_numRows = col.length;
	}
	
	/**
	 * Append a column of value type INT as the last column of 
	 * the data frame. The given array is wrapped but not copied 
	 * and hence might be updated in the future.
	 * 
	 * @param col
	 */
	public void appendColumn(long[] col) {
		ensureColumnCompatibility(col.length);
		_schema.add(ValueType.INT);
		_coldata.add(new LongArray(col));
		_numRows = col.length;
	}
	
	/**
	 * Append a column of value type DOUBLE as the last column of 
	 * the data frame. The given array is wrapped but not copied 
	 * and hence might be updated in the future.
	 * 
	 * @param col
	 */
	public void appendColumn(double[] col) {
		ensureColumnCompatibility(col.length);
		_schema.add(ValueType.DOUBLE);
		_coldata.add(new DoubleArray(col));
		_numRows = col.length;
	}
	
	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as strings independent of their value types.  
	 * 
	 * @return
	 */
	public Iterator<String[]> getStringRowIterator() {
		return new StringRowIterator();
	}
	
	/**
	 * Get a row iterator over the frame where all fields are encoded
	 * as boxed objects according to their value types.  
	 * 
	 * @return
	 */
	public Iterator<Object[]> getObjectRowIterator() {
		return new ObjectRowIterator();
	}

	///////
	// serialization / deserialization (implementation of writable and externalizable)
	
	@Override
	public void write(DataOutput out) throws IOException {
		//write header (rows, cols)
		out.writeInt(getNumRows());
		out.writeInt(getNumColumns());
		//write columns (value type, data)
		for( int j=0; j<getNumColumns(); j++ ) {
			out.writeByte(_schema.get(j).ordinal());
			out.writeUTF(_colnames.get(j));
			_coldata.get(j).write(out);
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		//read head (rows, cols)
		_numRows = in.readInt();
		int numCols = in.readInt();
		//read columns (value type, data)
		_schema.clear();
		_coldata.clear();
		for( int j=0; j<numCols; j++ ) {
			ValueType vt = ValueType.values()[in.readByte()];
			String name = in.readUTF();
			Array arr = null;
			switch( vt ) {
				case STRING:  arr = new StringArray(new String[_numRows]); break;
				case BOOLEAN: arr = new BooleanArray(new boolean[_numRows]); break;
				case INT:     arr = new LongArray(new long[_numRows]); break;
				case DOUBLE:  arr = new DoubleArray(new double[_numRows]); break;
				default: throw new IOException("Unsupported value type: "+vt);
			}
			arr.readFields(in);
			_schema.add(vt);
			_colnames.add(name);
			_coldata.add(arr);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		//redirect serialization to writable impl
		write(out);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		//redirect deserialization to writable impl
		readFields(in);
	}
	
	////////
	// CacheBlock implementation
	
	public long getInMemorySize() {
		return 1;
	}
	
	@Override
	public long getExactSerializedSize() {
		//TODO implement getExactSizeOnDisk();
		return 1;
	}
	
	@Override
	public boolean isShallowSerialize() {
		//shallow serialize since frames always dense
		return true;
	}
	
	@Override
	public void compactEmptyBlock() {
		//do nothing
	}
	
	///////
	// indexing and append operations
	
	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, IndexRange ixrange, FrameBlock ret)
		throws DMLRuntimeException
	{
		return leftIndexingOperations(rhsFrame, 
				(int)ixrange.rowStart, (int)ixrange.rowEnd, 
				(int)ixrange.colStart, (int)ixrange.colEnd, ret);
	}
	
	/**
	 * 
	 * @param rhsFrame
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param ret
	 * @return
	 */
	public FrameBlock leftIndexingOperations(FrameBlock rhsFrame, int rl, int ru, int cl, int cu, FrameBlock ret)
		throws DMLRuntimeException
	{
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
		ret._schema = new ArrayList<ValueType>(_schema);
		ret._colnames = new ArrayList<String>(_colnames);
		
		//copy data to output and partial overwrite w/ rhs
		for( int j=0; j<getNumColumns(); j++ ) {
			Array tmp = _coldata.get(j).clone();
			if( j>=cl && j<=cu )
				tmp.set(rl, ru, rhsFrame._coldata.get(j-cl));
			ret._coldata.add(tmp);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param ixrange
	 * @param ret
	 * @return
	 * @throws DMLRuntimeException
	 */
	public FrameBlock sliceOperations(IndexRange ixrange, FrameBlock ret) 
		throws DMLRuntimeException
	{
		return sliceOperations(
				(int)ixrange.rowStart, (int)ixrange.rowEnd,
				(int)ixrange.colStart, (int)ixrange.colEnd, ret);
	}
	
	/**
	 * Right indexing operations to slice a subframe out of this frame block. 
	 * Note that the existing column value types are preserved.
	 * 
	 * @param rl row lower index, inclusive, 0-based
	 * @param ru row upper index, inclusive, 0-based
	 * @param cl column lower index, inclusive, 0-based
	 * @param cu column upper index, inclusive, 0-based
	 * @param ret
	 * @return
	 */
	public FrameBlock sliceOperations(int rl, int ru, int cl, int cu, FrameBlock ret) 
		throws DMLRuntimeException
	{
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
			ret.reset(ru-rl+1);
		
		//copy output schema and colnames
		for( int j=cl; j<=cu; j++ ) {
			ret._schema.add(_schema.get(j));
			ret._colnames.add(_colnames.get(j));
		}	
		ret._numRows = ru-rl+1;

		//copy output data
		if(ret._coldata.size() == 0)
			for( int j=cl; j<=cu; j++ )
				ret._coldata.add(_coldata.get(j).slice(rl,ru));
		else
			for( int j=cl; j<=cu; j++ )
				ret._coldata.get(j-cl).set(0, ru-rl+1, _coldata.get(j), rl);	

		return ret;
	}
	
	/**
	 * Appends the given argument frameblock 'that' to this frameblock by 
	 * creating a deep copy to prevent side effects. For cbind, the frames
	 * are appended column-wise (same number of rows), while for rbind the 
	 * frames are appended row-wise (same number of columns).   
	 * 
	 * @param that
	 * @param ret
	 * @param cbind
	 * @return
	 */
	public FrameBlock appendOperations( FrameBlock that, FrameBlock ret, boolean cbind )
		throws DMLRuntimeException
	{
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
			ret._schema = new ArrayList<ValueType>(_schema);
			ret._schema.addAll(that._schema);
			ret._colnames = new ArrayList<String>(_colnames);
			ret._colnames.addAll(that._colnames);
			
			//concatenate column data (w/ deep copy to prevent side effects)
			for( Array tmp : _coldata )
				ret._coldata.add(tmp.clone());
			for( Array tmp : that._coldata )
				ret._coldata.add(tmp.clone());	
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
			ret._schema = new ArrayList<ValueType>(_schema);
			ret._colnames = new ArrayList<String>(_colnames);
			
			//concatenate data (deep copy first, append second)
			for( Array tmp : _coldata )
				ret._coldata.add(tmp.clone());
			Iterator<Object[]> iter = that.getObjectRowIterator();
			while( iter.hasNext() )
				ret.appendRow(iter.next());
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param src
	 * @throws DMLRuntimeException 
	 */
	public void copy(FrameBlock src) {
		copy(0, src.getNumRows()-1, 0, src.getNumColumns()-1, src);
	}

	/**
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param src
	 */
	public void copy(int rl, int ru, int cl, int cu, FrameBlock src) 
	{
		//allocate columns if necessary
		ensureAllocatedColumns(ru-rl+1);
		
		//copy values
		for( int j=cl; j<=cu; j++ ) {
			//special case: column memcopy 
			if( _schema.get(j).equals(src._schema.get(j-cl)) )
				_coldata.get(j).set(rl, ru, src._coldata.get(j-cl));
			//general case w/ schema transformation
			else 
				for( int i=rl; i<=ru; i++ ) {
					String tmp = src.get(i-rl, j)!=null ? src.get(i-rl, j).toString() : null;
					set(i, j, UtilFunctions.stringToObject(_schema.get(j), tmp));
				}
		}
	}
	
	
	///////
	// transform specific functionality
	
	/**
	 * 
	 * @param col
	 * @return
	 */
	public HashMap<String,Long> getRecodeMap(int col) {
		//probe cache for existing map
		if( REUSE_RECODE_MAPS ) {
			SoftReference<HashMap<String,Long>> tmp = _rcdMapCache.get(col);
			HashMap<String,Long> map = (tmp!=null) ? tmp.get() : null;
			if( map != null ) return map;
		}
		
		//construct recode map
		HashMap<String,Long> map = new HashMap<String,Long>();
		Array ldata = _coldata.get(col); 
		for( int i=0; i<getNumRows(); i++ ) {
			Object val = ldata.get(i);
			if( val != null ) {
				String[] tmp = val.toString().split(Lop.DATATYPE_PREFIX);
				map.put(tmp[0], Long.parseLong(tmp[1]));
			}
		}
		
		//put created map into cache
		if( REUSE_RECODE_MAPS ) {
			_rcdMapCache.put(col, new SoftReference<HashMap<String,Long>>(map));
		}
		
		return map;
	}

	
	///////
	// row iterators (over strings and boxed objects)
	
	/**
	 * 
	 */
	private abstract class RowIterator<T> implements Iterator<T[]> {
		protected T[] _curRow = null;
		protected int _curPos = -1;
		
		protected RowIterator() {
			_curPos = 0;
			_curRow = createRow(getNumColumns());
		}
		
		@Override
		public boolean hasNext() {
			return (_curPos < _numRows);
		}

		@Override
		public void remove() {
			throw new RuntimeException("RowIterator.remove is unsupported!");			
		}
		
		protected abstract T[] createRow(int size);
	}
	
	/**
	 * 
	 */
	private class StringRowIterator extends RowIterator<String> {
		@Override
		protected String[] createRow(int size) {
			return new String[size];
		}
		
		@Override
		public String[] next( ) {
			for( int j=0; j<getNumColumns(); j++ ) {
				Object tmp = get(_curPos, j);
				_curRow[j] = (tmp!=null) ? tmp.toString() : null;
			}
			_curPos++;			
			return _curRow;
		}
	}
	
	/**
	 * 
	 */
	private class ObjectRowIterator extends RowIterator<Object> {
		@Override
		protected Object[] createRow(int size) {
			return new Object[size];
		}
		
		@Override
		public Object[] next( ) {
			for( int j=0; j<getNumColumns(); j++ )
				_curRow[j] = get(_curPos, j);
			_curPos++;			
			return _curRow;
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
		protected int _size = 0;
		protected int newSize() {
			return (int) Math.max(_size*2, 4); 
		}
		public abstract T get(int index);
		public abstract void set(int index, T value);
		public abstract void set(int rl, int ru, Array value);
		public abstract void set(int rl, int ru, Array value, int rlSrc);
		public abstract void append(String value);
		public abstract void append(T value);
		public abstract Array clone();
		public abstract Array slice(int rl, int ru);
	}
	
	/**
	 * 
	 */
	private static class StringArray extends Array<String> {
		private String[] _data = null;
		
		public StringArray(String[] data) {
			_data = data;
			_size = _data.length;
		}		
		public String get(int index) {
			return _data[index];
		}
		public void set(int index, String value) {
			_data[index] = value;
		}
		public void set(int rl, int ru, Array value) {
			System.arraycopy(((StringArray)value)._data, 0, _data, rl, ru-rl+1);
		}
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((StringArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		public void append(String value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
		}
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeUTF((_data[i]!=null)?_data[i]:"");
		}
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ ) {
				String tmp = in.readUTF();
				_data[i] = (!tmp.isEmpty()) ? tmp : null;
			}
		}
		public Array clone() {
			return new StringArray(Arrays.copyOf(_data, _size));
		}
		public Array slice(int rl, int ru) {
			return new StringArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
	}
	
	/**
	 * 
	 */
	private static class BooleanArray extends Array<Boolean> {
		private boolean[] _data = null;
		
		public BooleanArray(boolean[] data) {
			_data = data;
			_size = _data.length;
		}		
		public Boolean get(int index) {
			return _data[index];
		}
		public void set(int index, Boolean value) {
			_data[index] = (value!=null) ? value : false;
		}
		public void set(int rl, int ru, Array value) {
			System.arraycopy(((BooleanArray)value)._data, 0, _data, rl, ru-rl+1);
		}
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((BooleanArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		public void append(String value) {
			append(Boolean.parseBoolean(value));
		}
		public void append(Boolean value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : false;
		}
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeBoolean(_data[i]);
		}
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readBoolean();
		}
		public Array clone() {
			return new BooleanArray(Arrays.copyOf(_data, _size));
		}
		public Array slice(int rl, int ru) {
			return new BooleanArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
	}
	
	/**
	 * 
	 */
	private static class LongArray extends Array<Long> {
		private long[] _data = null;
		
		public LongArray(long[] data) {
			_data = data;
			_size = _data.length;
		}		
		public Long get(int index) {
			return _data[index];
		}
		public void set(int index, Long value) {
			_data[index] = (value!=null) ? value : 0L;
		}
		public void set(int rl, int ru, Array value) {
			System.arraycopy(((LongArray)value)._data, 0, _data, rl, ru-rl+1);
		}
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((LongArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		public void append(String value) {
			append((value!=null)?Long.parseLong(value):null);
		}
		public void append(Long value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0L;
		}
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeLong(_data[i]);
		}
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readLong();
		}
		public Array clone() {
			return new LongArray(Arrays.copyOf(_data, _size));
		}
		public Array slice(int rl, int ru) {
			return new LongArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
	}
	
	/**
	 * 
	 */
	private static class DoubleArray extends Array<Double> {
		private double[] _data = null;
		
		public DoubleArray(double[] data) {
			_data = data;
			_size = _data.length;
		}		
		public Double get(int index) {
			return _data[index];
		}
		public void set(int index, Double value) {
			_data[index] = (value!=null) ? value : 0d;
		}
		public void set(int rl, int ru, Array value) {
			System.arraycopy(((DoubleArray)value)._data, 0, _data, rl, ru-rl+1);
		}
		public void set(int rl, int ru, Array value, int rlSrc) {
			System.arraycopy(((DoubleArray)value)._data, rlSrc, _data, rl, ru-rl+1);
		}
		public void append(String value) {
			append((value!=null)?Double.parseDouble(value):null);
		}
		public void append(Double value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = (value!=null) ? value : 0d;
		}
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeDouble(_data[i]);
		}
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readDouble();
		}
		public Array clone() {
			return new DoubleArray(Arrays.copyOf(_data, _size));
		}
		public Array slice(int rl, int ru) {
			return new DoubleArray(Arrays.copyOfRange(_data,rl,ru+1));
		}
	}
}
