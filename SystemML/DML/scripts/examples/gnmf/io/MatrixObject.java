package gnmf.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class MatrixObject implements WritableComparable<MatrixObject>
{
	public static final int OBJECT_TYPE_CELL			= 0;
	public static final int OBJECT_TYPE_VECTOR			= 1;
	public static final int OBJECT_TYPE_SQUAREMATRIX	= 2;
	public static final int OBJECT_TYPE_MATRIX			= 3;
	private int objectType;
	MatrixFormats containedObject;
	
	
	public MatrixObject()
	{
		
	}
	
	public MatrixObject(MatrixCell cell)
	{
		objectType = OBJECT_TYPE_CELL;
		containedObject = cell;
	}
	
	public MatrixObject(MatrixVector vector)
	{
		objectType = OBJECT_TYPE_VECTOR;
		containedObject = vector;
	}
	
	public MatrixObject(SquareMatrix matrix)
	{
		objectType = OBJECT_TYPE_SQUAREMATRIX;
		containedObject = matrix;
	}
	
	public MatrixObject(Matrix matrix)
	{
		objectType = OBJECT_TYPE_MATRIX;
		containedObject = matrix;
	}
	
	public MatrixFormats getObject()
	{
		return containedObject;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException
	{
		objectType = in.readInt();
		switch(objectType)
		{
		case OBJECT_TYPE_CELL:
			containedObject = new MatrixCell();
			break;
		case OBJECT_TYPE_VECTOR:
			containedObject = new MatrixVector();
			break;
		case OBJECT_TYPE_MATRIX:
			containedObject = new Matrix();
			break;
		case OBJECT_TYPE_SQUAREMATRIX:
			containedObject = new SquareMatrix();
			break;
		}
		containedObject.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		out.writeInt(objectType);
		containedObject.write(out);
	}

	@Override
	public int compareTo(MatrixObject that)
	{
		return 0;
	}
}
