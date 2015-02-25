/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;


import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import com.ibm.bi.dml.runtime.util.UtilFunctions;



public class BinaryBlockToTextCellConverter implements 
Converter<MatrixIndexes, MatrixBlock, NullWritable, Text>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private SparseRowsIterator sparseIterator=null;
	private double[] denseArray=null;
	private int denseArraySize=0;
	private int nextInDenseArray=-1;
	private boolean sparse=true;
	private int thisBlockWidth=0;
	private MatrixIndexes startIndexes=new MatrixIndexes();
	private boolean hasValue=false;
	private int brow;
	private int bcolumn;
	
	private Text value=new Text();
	private Pair<NullWritable, Text> pair=new Pair<NullWritable, Text>(NullWritable.get(), value);
	
	private void reset()
	{
		sparseIterator=null;
		denseArray=null;
		denseArraySize=0;
		nextInDenseArray=-1;
		sparse=true;
		thisBlockWidth=0;
	}
	
	@Override
	public void convert(MatrixIndexes k1, MatrixBlock v1) {
		reset();
		startIndexes.setIndexes(UtilFunctions.cellIndexCalculation(k1.getRowIndex(), brow,0), 
				UtilFunctions.cellIndexCalculation(k1.getColumnIndex(),bcolumn,0));
		sparse=v1.isInSparseFormat();
		thisBlockWidth=v1.getNumColumns();
		if(sparse)
		{
			sparseIterator=v1.getSparseRowsIterator();
		}
		else
		{
			if(v1.getDenseArray()==null)
				return;
			denseArray=v1.getDenseArray();
			nextInDenseArray=0;
			denseArraySize=v1.getNumRows()*v1.getNumColumns();
		}
		hasValue=(v1.getNonZeros()>0);
	}

	@Override
	public boolean hasNext() {
		if(sparse)
		{
			if(sparseIterator==null)
				hasValue=false;
			else
				hasValue=sparseIterator.hasNext();
		}else
		{
			if(denseArray==null)
				hasValue=false;
			else
			{
				while(nextInDenseArray<denseArraySize && denseArray[nextInDenseArray]==0)
					nextInDenseArray++;
				hasValue=(nextInDenseArray<denseArraySize);
			}
		}
		return hasValue;
	}

	@Override
	public Pair<NullWritable, Text> next() {
		if(!hasValue)
			return null;
		long i, j;
		double v;
		if(sparse)
		{
			if(sparseIterator==null)
				return null;
			else
			{
				IJV cell = sparseIterator.next();
				i = cell.i + startIndexes.getRowIndex();
				j = cell.j + startIndexes.getColumnIndex();
				v = cell.v;
			}
				
		}else
		{
			if(denseArray==null)
				return null;
			else
			{
				i=startIndexes.getRowIndex() + nextInDenseArray/thisBlockWidth;
				j=startIndexes.getColumnIndex() + nextInDenseArray%thisBlockWidth;
				v=denseArray[nextInDenseArray];
				nextInDenseArray++;
			}
		}
		value.set(i+" "+j+" "+v);
		return pair;
	}

	public void setBlockSize(int nr, int nc) {
		brow=nr;
		bcolumn=nc;
	}
}
