package gnmf.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class MatrixCell implements MatrixFormats
{
	private long column;
	private double value;
	
	
	public MatrixCell()
	{
		
	}
	
	public MatrixCell(long column, double value)
	{
		this.column	= column;
		this.value	= value;
	}
	
	public long getColumn()
	{
		return column;
	}
	
	public double getValue()
	{
		return value;
	}

	@Override
	public void readFields(DataInput in) throws IOException
	{
		column	= in.readLong();
		value	= in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		out.writeLong(column);
		out.writeDouble(value);
	}

	@Override
	public int compareTo(MatrixFormats that)
	{
		return 0;
	}
}
