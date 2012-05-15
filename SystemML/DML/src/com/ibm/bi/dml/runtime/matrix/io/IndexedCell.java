package com.ibm.bi.dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.mapred.CachedMapElement;



public class IndexedCell extends CachedMapElement implements Writable{
	
	public IndexedCell(){};
	public IndexedCell(long r, long c, double v)
	{
		set(r,c,v);
	}
	public void set(long r, long c, double v)
	{
		row=r;
		column=c;
		value=v;
	}
	
	public void set(MatrixIndexes indexes, DoubleWritable v)
	{
		set(indexes.getRowIndex(), indexes.getColumnIndex(), v.get());
	}
	
	public String toString()
	{
		return "("+row+", "+column+"): "+value;
	}
	
	public long getRowIndex()
	{
		return row;
	}
	public long getColumnIndex()
	{
		return column;
	}
	public double getValue()
	{
		return value;
	}
	
	private long row=-1;
	private long column=-1;
	private double value=0;
	
	@Override
	public void readFields(DataInput in) throws IOException {
		row=in.readLong();
		column=in.readLong();
		value=in.readDouble();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(row);
		out.writeLong(column);
		out.writeDouble(value);
	}
	
	public void setValue(double v) {
		value=v;
	}
	public void setIndexes(MatrixIndexes indexes) {
		row=indexes.getRowIndex();
		column=indexes.getColumnIndex();
	}
	public void setIndexes(long r, long c)
	{
		row=r;
		column=c;
	}
	@Override
	public CachedMapElement duplicate() {
		return new IndexedCell(row, column, value);
	}
	@Override
	public void set(CachedMapElement elem) {
		if(elem instanceof IndexedCell)
		{
			IndexedCell that=(IndexedCell) elem;
			this.row=that.row;
			this.column=that.column;
			this.value=that.value;
		}
	}

}
