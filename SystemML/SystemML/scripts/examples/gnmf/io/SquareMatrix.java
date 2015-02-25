package gnmf.io;

import java.io.Serializable;

public class SquareMatrix extends Matrix implements Serializable
{
	private static final long serialVersionUID = -7900756372081924705L;
	

	public SquareMatrix()
	{
		
	}
	
	public SquareMatrix(long size, double[] values)
	{
		super(size, size, values);
	}
	
	public SquareMatrix(MatrixVector vector)
	{
		this(vector.getValues());
	}
	
	public void addMatrix(SquareMatrix matrix2)
	{
		for(long r = 0; r < rows; r++)
		{
			for(long c = 0; c < columns; c++)
			{
				values[(int) (r * columns + c)] += matrix2.getValue(r, c);
			}
		}
	}
	
	public MatrixVector multiplyWithVectorFirst(MatrixVector vector)
	{
		double[] vectorValues = vector.getValues();
		double[] resultValues = new double[vectorValues.length];
		for(long c = 0; c < columns; c++)
		{
			double sum = 0.0;
			for(long r = 0; r < rows; r++)
			{
				sum += (values[(int) (r * columns + c)] * vectorValues[(int) r]);
			}
			resultValues[(int) c] = sum;
		}
		return new MatrixVector(resultValues);
	}
	
	public MatrixVector multiplyWithVectorSecond(MatrixVector vector)
	{
		double[] vectorValues = vector.getValues();
		double[] resultValues = new double[vectorValues.length];
		for(long r = 0; r < rows; r++)
		{
			double sum = 0.0;
			for(long c = 0; c < columns; c++)
			{
				sum += (values[(int) (r * columns + c)] * vectorValues[(int) c]);
			}
			resultValues[(int) r] = sum;
		}
		return new MatrixVector(resultValues);
	}
	
	public SquareMatrix(double[] vectorValues)
	{
		rows	= vectorValues.length;
		columns	= vectorValues.length;
		values	= new double[(int) (rows * columns)];
		for(int r = 0; r < rows; r++)
		{
			for(int c = 0; c < columns; c++)
			{
				values[(int) (r * columns + c)] = vectorValues[r] * vectorValues[c];
			}
		}
	}
	
	public SquareMatrix getCopy()
	{
		return new SquareMatrix(rows, values);
	}
}
