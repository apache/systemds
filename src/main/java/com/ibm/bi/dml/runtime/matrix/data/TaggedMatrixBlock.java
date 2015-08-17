/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

public class TaggedMatrixBlock extends TaggedMatrixValue
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public TaggedMatrixBlock(MatrixBlock b, byte t) {
		super(b, t);
	}

	public TaggedMatrixBlock()
	{        
        tag=-1;
     	base=new MatrixBlock();
	}

	public TaggedMatrixBlock(TaggedMatrixBlock that) {
		tag=that.getTag();
		base=new MatrixBlock();
		base.copy(that.getBaseObject());
	}
}