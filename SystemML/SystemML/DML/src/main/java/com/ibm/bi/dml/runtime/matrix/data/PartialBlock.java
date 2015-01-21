/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class PartialBlock implements WritableComparable<PartialBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
