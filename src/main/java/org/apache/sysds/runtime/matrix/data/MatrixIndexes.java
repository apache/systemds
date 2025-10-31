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

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * This represent the indexes to the blocks of the matrix.
 * Please note that these indexes are 1-based, whereas the data in the block are zero-based (as they are double arrays).
 */
public class MatrixIndexes implements WritableComparable<MatrixIndexes>, RawComparator<MatrixIndexes>, Externalizable
{	
	private static final long serialVersionUID = -1521166657518127789L;
	
	private long _row = -1;
	private long _col = -1;
	
	///////////////////////////
	// constructors
	
	public MatrixIndexes() {
		//do nothing
	}
	
	public MatrixIndexes(long r, long c) {
		setIndexes(r,c);
	}
	
	public MatrixIndexes(MatrixIndexes indexes) {
		setIndexes(indexes._row, indexes._col);
	}
	
	///////////////////////////
	// get/set methods

	public long getRowIndex() {
		return _row;
	}
	
	public long getColumnIndex() {
		return _col;
	}
	
	public MatrixIndexes setIndexes(long r, long c) {
		_row = r;
		_col = c;
		return this;
	}
	
	public MatrixIndexes setIndexes(MatrixIndexes that) {
		_row = that._row;
		_col = that._col;
		return this;
	}
	
	@Override
	public int compareTo(MatrixIndexes other) {
		if( _row != other._row )
			return (_row > other._row ? 1 : -1);
		else if( _col != other._col)
			return (_col > other._col ? 1 : -1);
		return 0;
	}

	@Override
	public boolean equals(Object other) {
		if( !(other instanceof MatrixIndexes))
			return false;
		
		MatrixIndexes tother = (MatrixIndexes)other;
		return (_row==tother._row && _col==tother._col);
	}
	
	@Override
	public int hashCode() {
		return UtilFunctions.longHashCode(_row, _col);
	}
	
	@Override
	public String toString() {
		return "("+_row+", "+_col+")";
	}
	
	public MatrixIndexes fromString(String ix) {
		String[] parts = ix.substring(1, ix.length()-1).split(",");
		return new MatrixIndexes(Long.parseLong(parts[0]), Long.parseLong(parts[1].trim()));
	}

	////////////////////////////////////////////////////
	// implementation of Writable read/write

	@Override
	public void readFields(DataInput in) throws IOException {
		_row = in.readLong();
		_col = in.readLong();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong( _row );
		out.writeLong( _col );
	}
	
	
	////////////////////////////////////////////////////
	// implementation of Externalizable read/write

	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for consistency/maintainability. 
	 * 
	 * @param is object input
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void readExternal(ObjectInput is) 
		throws IOException
	{
		//default deserialize (general case)
		readFields(is);
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for consistency/maintainability. 
	 * 
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput os) 
		throws IOException
	{
		//default serialize (general case)
		write(os);	
	}
	
	////////////////////////////////////////////////////
	// implementation of RawComparator<MatrixIndexes>
	
	@Override
	public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2)
	{
		//compare row
		long v1 = WritableComparator.readLong(b1, s1);
	    long v2 = WritableComparator.readLong(b2, s2);
	    if(v1!=v2)
	    	return v1<v2 ? -1 : 1;    
	    //compare column (if needed)
		v1 = WritableComparator.readLong(b1, s1+Long.SIZE/8);
		v2 = WritableComparator.readLong(b2, s2+Long.SIZE/8);
		return (v1<v2 ? -1 : (v1==v2 ? 0 : 1));
	}

	@Override
	public int compare(MatrixIndexes m1, MatrixIndexes m2) {
		return m1.compareTo(m2);
	}
}
