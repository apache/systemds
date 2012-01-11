package dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

public class PartialBlock implements WritableComparable<PartialBlock>{

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

}
