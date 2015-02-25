package gnmf.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class MatrixVector implements MatrixFormats
{
	private double[] values;
	
	
	public MatrixVector()
	{
		
	}
	
	public MatrixVector(double[] values)
	{
		this.values = values.clone();
	}
	
	public MatrixVector(int k)
	{
		values = new double[k];
	}
	
	public double[] getValues()
	{
		return values;
	}
	
	public MatrixVector getCopy()
	{
		return new MatrixVector(values);
	}
	
	public void multiplyWithScalar(double scalar)
	{
		for(int i = 0; i < values.length; i++)
		{
			values[i] *= scalar;
		}
	}
	
	public void addVector(MatrixVector vector2)
	{
		double[] vector2Values = vector2.getValues();
		if(values.length != vector2Values.length)
			throw new IllegalArgumentException("both vectors have to contain the same " +
					"number of elements (" + values.length + " | " + vector2Values.length + ")");
		
		for(int i = 0; i < values.length; i++)
		{
			values[i] += vector2Values[i];
		}
	}
	
	public void elementWiseMultiplication(MatrixVector vector2)
	{
		double[] vector2Values = vector2.getValues();
		for(int i = 0; i < values.length; i++)
		{
			values[i] *= vector2Values[i];
		}
	}
	
	public void elementWiseDivision(MatrixVector vector2)
	{
		double[] vector2Values = vector2.getValues();
		for(int i = 0; i < values.length; i++)
		{
			values[i] /= vector2Values[i];
		}
	}
	
	@Override
	public void readFields(DataInput in) throws IOException
	{
		int size = in.readInt();
		values = new double[size];
		for(int i = 0; i < size; i++)
		{
			values[i] = in.readDouble();
		}
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		out.writeInt(values.length);
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
