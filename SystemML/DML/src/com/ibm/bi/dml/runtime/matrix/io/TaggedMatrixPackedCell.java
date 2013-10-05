/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

public class TaggedMatrixPackedCell extends TaggedMatrixValue
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public TaggedMatrixPackedCell(MatrixPackedCell b, byte t)
	{
		super(b, t);
	}
	public TaggedMatrixPackedCell()
	{
		super();
        tag=-1;
     	base=new MatrixPackedCell();
	}
	public TaggedMatrixPackedCell(TaggedMatrixPackedCell that)
	{
		this.tag=that.tag;
		base=new MatrixPackedCell((MatrixPackedCell) that.base);
	}
}