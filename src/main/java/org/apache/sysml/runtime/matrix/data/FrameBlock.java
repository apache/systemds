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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * 
 */
@SuppressWarnings({"rawtypes","unchecked"}) //allow generic native arrays
public class FrameBlock implements Writable, Externalizable
{
	private static final long serialVersionUID = -3993450030207130665L;
	
	/** The number of rows of the FrameBlock */
	private int _numRows = -1;
	
	/** The schema of the data frame as an ordered list of value types */
	private List<ValueType> _schema = null; 
	
	/** The data frame data as an ordered list of columns */
	private List<Array> _coldata = null;
	
	public FrameBlock() {
		_schema = new ArrayList<ValueType>();
		_coldata = new ArrayList<Array>();
	}
	
	public FrameBlock(List<ValueType> schema) {
		this(schema, new String[0][]);
	}
	
	public FrameBlock(List<ValueType> schema, String[][] data) {
		_numRows = data.length;
		_schema = new ArrayList<ValueType>(schema);
		_coldata = new ArrayList<Array>();
		for( int i=0; i<data.length; i++ )
			appendRow(data[i]);
	}
	
	/**
	 * Get the number of rows of the data frame.
	 * 
	 * @return 
	 */
	public int getNumRows() {
		return _numRows;
	}
	
	/**
	 * Get the number of columns of the data frame, that is
	 * the number of columns defined in the schema.
	 * 
	 * @return
	 */
	public int getNumColumns() {
		return _schema.size();
	}
	
	/**
	 * Allocate column data structures if necessary, i.e., if schema specified
	 * but not all column data structures created yet.
	 */
	public void ensureAllocatedColumns() {
		//early abort if already 
		if( _schema.size() == _coldata.size() ) 
			return;		
		//allocate columns if necessary
		for( int j=0; j<_schema.size(); j++ ) {
			if( j >= _coldata.size() )
				switch( _schema.get(j) ) {
					case STRING:  _coldata.add(new StringArray(new String[0])); break;
					case BOOLEAN: _coldata.add(new BooleanArray(new boolean[0])); break;
					case INT:     _coldata.add(new LongArray(new long[0])); break;
					case DOUBLE:  _coldata.add(new DoubleArray(new double[0])); break;
					default: throw new RuntimeException("Unsupported value type: "+_schema.get(j));
				}
		}
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
	
	/**
	 * Append a row to the end of the data frame, where all row fields
	 * are boxed objects according to the schema.
	 * 
	 * @param row
	 */
	public void appendRow(Object[] row) {
		ensureAllocatedColumns();
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
		ensureAllocatedColumns();
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
			for( int j=0; j<getNumColumns(); j++ )
				_curRow[j] = get(_curPos, j).toString();
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
		public abstract void append(String value);
		public abstract void append(T value);
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
		public void append(String value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
		}
		@Override
		public void write(DataOutput out) throws IOException {
			for( int i=0; i<_size; i++ )
				out.writeUTF(_data[i]);
		}
		@Override
		public void readFields(DataInput in) throws IOException {
			_size = _data.length;
			for( int i=0; i<_size; i++ )
				_data[i] = in.readUTF();
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
			_data[index] = value;
		}
		public void append(String value) {
			append(Boolean.parseBoolean(value));
		}
		public void append(Boolean value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
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
			_data[index] = value;
		}
		public void append(String value) {
			append(Long.parseLong(value));
		}
		public void append(Long value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
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
			_data[index] = value;
		}
		public void append(String value) {
			append(Double.parseDouble(value));
		}
		public void append(Double value) {
			if( _data.length <= _size )
				_data = Arrays.copyOf(_data, newSize());
			_data[_size++] = value;
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
	}
}
