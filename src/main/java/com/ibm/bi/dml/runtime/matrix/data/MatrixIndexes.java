/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.RawComparator;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * This represent the indexes to the blocks of the matrix.
 * Please note that these indexes are 1-based, whereas the data in the block are zero-based (as they are double arrays).
 */
public class MatrixIndexes implements WritableComparable<MatrixIndexes>, RawComparator<MatrixIndexes>, Serializable
{
	
	private static final long serialVersionUID = -1521166657518127789L;
		
	public static final int BYTE_SIZE = (Long.SIZE+Long.SIZE)/8;
	public static final long ADD_PRIME1 = 99991;
	public static final long ADD_PRIME2 = 853;
	public static final int DIVIDE_PRIME = 1405695061; 
	//prime close to max int, because it determines the max hash domain size
	//public static final int DIVIDE_PRIME = 51473;
	
	private long row = -1;
	private long column = -1;
	
	///////////////////////////
	// constructors
	
	public MatrixIndexes(){
		//do nothing
	}
	
	public MatrixIndexes(long r, long c){
		setIndexes(r,c);
	}
	
	public MatrixIndexes(MatrixIndexes indexes) {
		setIndexes(indexes.row, indexes.column);
	}
	
	///////////////////////////
	// get/set methods

	
	public long getRowIndex() {
		return row;
	}
	
	public long getColumnIndex() {
		return column;
	}
	
	public void setIndexes(long r, long c) {
		row = r;
		column = c;
	}
	
	public void setIndexes(MatrixIndexes that) {
		
		this.row=that.row;
		this.column=that.column;
	}
	
	
	
	@Override
	public void readFields(DataInput in) throws IOException {
		row=in.readLong();
		column=in.readLong();
		
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(row);
		out.writeLong(column);
		
	}
	
	@Override
	public int compareTo(MatrixIndexes other)
	{
		if(this.row!=other.row)
			return (this.row>other.row? 1:-1);
		else if(this.column!=other.column)
			return (this.column>other.column? 1:-1);
		return 0;
	}

	@Override
	public boolean equals(Object other)
	{
		if( !(other instanceof MatrixIndexes))
			return false;
		
		MatrixIndexes tother = (MatrixIndexes)other;
		return (this.row==tother.row && this.column==tother.column);
	}
	
	 public int hashCode() {
		 return UtilFunctions.longHashFunc((row<<32)+column+ADD_PRIME1)%DIVIDE_PRIME;
	 }

	public void print() {
		System.out.println("("+row+", "+column+")");
	}
	
	public String toString()
	{
		return "("+row+", "+column+")";
	}
	
	public int compareWithOrder(MatrixIndexes other, boolean leftcached) {
		if(!leftcached)
			return compareTo(other);
		
		if(this.column!=other.column)
			return (this.column>other.column? 1:-1);
		else if(this.row!=other.row)
			return (this.row>other.row? 1:-1);
		return 0;
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
