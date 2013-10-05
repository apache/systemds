/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import com.ibm.bi.dml.runtime.matrix.WriteCSVMR.RowBlock;

public class BinaryCellToRowBlockConverter implements Converter<MatrixIndexes, MatrixCell, MatrixIndexes, RowBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes returnIndexes=new MatrixIndexes();
	private RowBlock rowBlock=new RowBlock();
	private Pair<MatrixIndexes, RowBlock> pair=new Pair<MatrixIndexes, RowBlock>(returnIndexes, rowBlock);
	private boolean hasValue=false;

	@Override
	public void convert(MatrixIndexes k1, MatrixCell v1) {
		double v=((MatrixCell)v1).getValue();
		returnIndexes.setIndexes(k1);
		if(rowBlock.container==null || rowBlock.container.length<1)
			rowBlock.container=new double[1];
		rowBlock.numCols=1;
		rowBlock.container[0]=v1.getValue();
		hasValue=true;
	}

	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, RowBlock> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	public static void main(String[] args) throws Exception {
		BinaryCellToRowBlockConverter conv=new BinaryCellToRowBlockConverter();
		conv.convert(new MatrixIndexes(1, 2), new MatrixCell(10));
		while(conv.hasNext())
		{
			Pair pair=conv.next();
			System.out.println(pair.getKey()+": "+pair.getValue());
		}
	}

	@Override
	public void setBlockSize(int rl, int cl) {
	}
}

