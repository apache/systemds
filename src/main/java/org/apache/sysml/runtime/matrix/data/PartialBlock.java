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

import org.apache.hadoop.io.WritableComparable;

public class PartialBlock implements WritableComparable<PartialBlock>
{
	
	private double value;
	private int row=-1;
	private int column=-1;
	
	public PartialBlock(int r, int c, double v)
	{
		set(r, c, v);
	}
	
	public PartialBlock(){}
	
	public void set(int r, int c, double v) {
		row=r;
		column=c;
		value=v;
	}

	public int getRowIndex()
	{
		return row;
	}
	
	public int getColumnIndex()
	{
		return column;
	}
	
	public double getValue()
	{
		return value;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		row=in.readInt();
		column=in.readInt();
		value=in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(row);
		out.writeInt(column);
		out.writeDouble(value);
	}

	public String toString()
	{
		return "["+row+", "+column+"]:"+value;
	}

	@Override
	public int compareTo(PartialBlock that) {
		if(row!=that.row)
			return row-that.row;
		else if(column!=that.column)
			return column-that.column;
		else return Double.compare(value, that.value);
	}

	@Override 
	public boolean equals(Object o) {
		if( !(o instanceof PartialBlock) )
			return false;
		
		PartialBlock that = (PartialBlock)o;
		return (row==that.row && column==that.column && value==that.value);
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
}
