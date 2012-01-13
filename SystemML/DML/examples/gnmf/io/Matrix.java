package gnmf.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Matrix implements MatrixFormats
{
	protected long rows;
	protected long columns;
	protected double[] values;
	
	
	public Matrix()
	{
		
	}
	
	public Matrix(long rows, long columns, double[] values)
	{
		this.rows		= rows;
		this.columns	= columns;
		this.values		= values.clone();
	}
	
	public double getValue(long row, long column)
	{
		return values[(int) (row * columns + column)];
	}
	
	public Matrix getCopy()
	{
		return new Matrix(rows, columns, values);
	}
	
	@Override
	public void readFields(DataInput in) throws IOException
	{
		rows = in.readLong();
		columns = in.readLong();
		values = new double[(int) (rows * columns)];
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				values[(int) (r * columns + c)] = in.readDouble();
			}
		}
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		out.writeLong(rows);
		out.writeLong(columns);
		for(double value : values)
		{
			out.writeDouble(value);
		}
	}

	@Override
	public int compareTo(MatrixFormats that)
	{
		return 0;
	}
}
