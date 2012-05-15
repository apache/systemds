package com.ibm.bi.dml.runtime.matrix.io;

public class TaggedMatrixValue extends Tagged<MatrixValue>{
	public TaggedMatrixValue(MatrixValue b, byte t) {
		super(b, t);
	}

	public TaggedMatrixValue() {
	}

	public static  TaggedMatrixValue createObject(Class<? extends MatrixValue> cls)
	{
		if(cls.equals(MatrixCell.class))
			return new TaggedMatrixCell();
		else if(cls.equals(MatrixPackedCell.class))
			return new TaggedMatrixPackedCell();
		else
			return new TaggedMatrixBlock();
	}
	
	public static  TaggedMatrixValue createObject(MatrixValue b, byte t)
	{
		if(b instanceof MatrixCell)
			return new TaggedMatrixCell((MatrixCell)b, t);
		else
			return new TaggedMatrixBlock((MatrixBlock)b, t);
	}
}