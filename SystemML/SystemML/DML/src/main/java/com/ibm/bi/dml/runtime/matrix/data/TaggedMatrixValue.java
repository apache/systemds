/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

public class TaggedMatrixValue extends Tagged<MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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