/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;


public class BinaryCellToRowBlockConverter implements Converter<MatrixIndexes, MatrixCell, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes returnIndexes=new MatrixIndexes();
	private MatrixBlock rowBlock = new MatrixBlock();
	private Pair<MatrixIndexes, MatrixBlock> pair=new Pair<MatrixIndexes, MatrixBlock>(returnIndexes, rowBlock);
	private boolean hasValue=false;

	@Override
	public void convert(MatrixIndexes k1, MatrixCell v1) 
	{
		returnIndexes.setIndexes(k1);
		rowBlock.reset(1, 1);
		rowBlock.quickSetValue(0, 0, v1.getValue());
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, MatrixBlock> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}
	
	@Override
	public void setBlockSize(int rl, int cl) {
	}
}

