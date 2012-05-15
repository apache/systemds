package com.ibm.bi.dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class BlockCell implements Writable{
	
	public BlockCell(){};
	public BlockCell(int r, int c, double v)
	{
		set(r,c,v);
	}
	public void set(int r, int c, double v)
	{
		row=r;
		column=c;
		value=v;
	}
	
	public String toString()
	{
		return "("+row+", "+column+"): "+value;
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
	
	private int row=-1;
	private int column=-1;
	private double value=0;
	
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
	
	public void setValue(double v) {
		value=v;
	}
	
	public void setIndexes(int r, int c)
	{
		row=r;
		column=c;
	}
}

