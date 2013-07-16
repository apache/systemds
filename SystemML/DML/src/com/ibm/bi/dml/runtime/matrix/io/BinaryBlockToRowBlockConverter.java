package com.ibm.bi.dml.runtime.matrix.io;

import java.util.Arrays;

import com.ibm.bi.dml.runtime.matrix.WriteCSVMR.RowBlock;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class BinaryBlockToRowBlockConverter implements 
Converter<MatrixIndexes, MatrixBlock, MatrixIndexes, RowBlock>{

	private SparseRow[] sparseRows=null;
	private double[] denseArray=null;
	private int currentRow=-1;//reuse currentPos for both sparse and dense
	private boolean sparse=true;
	private int numColsInSource=0;
	private int numRowsInSource=0;
	private int rowBlockingFactor;
	private boolean hasValue=false;
	private long startRowID=-1;
	private long colID=-1;
	
	private MatrixIndexes returnIndexes=new MatrixIndexes();
	private RowBlock rowBlock=new RowBlock();
	private Pair<MatrixIndexes, RowBlock> pair=new Pair<MatrixIndexes, RowBlock>(returnIndexes, rowBlock);
	
	private void reset()
	{
		sparseRows=null;
		denseArray=null;
		currentRow=-1;
		sparse=true;
		numColsInSource=0;
		numRowsInSource=0;
		startRowID=-1;
		colID=-1;
	}
	
	@Override
	public void convert(MatrixIndexes k1, MatrixBlock v1) {
		reset();
		numColsInSource=v1.getNumColumns();
		numRowsInSource=v1.getNumRows();
		startRowID=UtilFunctions.cellIndexCalculation(k1.getRowIndex(), rowBlockingFactor,0);
		colID=k1.getColumnIndex();
		sparse=v1.isInSparseFormat();
		if(rowBlock.container==null || rowBlock.container.length<numColsInSource)
			rowBlock.container=new double[numColsInSource];
		rowBlock.numCols=numColsInSource;
		
		if(sparse)
		{
			if(v1.getSparseRows()==null)
				return;
			sparseRows=v1.getSparseRows();
		}
		else
		{
			if(v1.getDenseArray()==null)
				return;
			denseArray=v1.getDenseArray();	
		}
		currentRow=0;
		hasValue=(v1.getNonZeros()>0);
	}

	@Override
	public boolean hasNext() {
		if(currentRow>=numRowsInSource)
			return false;
		if(sparse)
		{
			if(sparseRows==null || currentRow>=sparseRows.length)
				hasValue=false;
			else
			{
				while(currentRow<numRowsInSource && currentRow<sparseRows.length)
				{
					if(sparseRows[currentRow]==null || sparseRows[currentRow].size()==0)
						currentRow++;
					else
						break;
				}
				if(currentRow>=numRowsInSource || currentRow>=sparseRows.length)
					return false;
				else
					return true;
			}
		}else
		{
			if(denseArray==null)
				hasValue=false;
		}
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, RowBlock> next() {
		if(!hasValue || currentRow>=numRowsInSource)
			return null;
		
		returnIndexes.setIndexes(startRowID+currentRow, colID);
		if(sparse)
		{
			if(sparseRows==null || currentRow>=sparseRows.length)
				return null;
			else
			{
				Arrays.fill(rowBlock.container, 0, numColsInSource, 0);
				double[] vals=sparseRows[currentRow].getValueContainer();
				int[] cols=sparseRows[currentRow].getIndexContainer();
				for(int i=0; i<sparseRows[currentRow].size(); i++)
					rowBlock.container[cols[i]]=vals[i];
			}
				
		}else
		{
			if(denseArray==null)
				return null;
			else
				System.arraycopy(denseArray, currentRow*numColsInSource, rowBlock.container, 0, numColsInSource);
		}
		currentRow++;
		return pair;
	}

	public void setBlockSize(int nr, int nc) {
		rowBlockingFactor=nr;
	}
	
	public static void main(String[] args) throws Exception {
		
		MatrixBlock m1=new MatrixBlock(3, 2, true);
		//m1.setValue(0, 0, 1);
		//m1.setValue(0, 1, 2);
		m1.setValue(1, 0, 3);
		//m1.setValue(1, 1, 4);
	//	m1.setValue(2, 0, 5);
		//m1.setValue(2, 1, 6);
		System.out.println("matrix m1: ");
		m1.print();
		
		MatrixIndexes ind=new MatrixIndexes(10, 10);
		
		BinaryBlockToRowBlockConverter conv=new BinaryBlockToRowBlockConverter();
		conv.setBlockSize(3, 2);
		conv.convert(ind, m1);
		while(conv.hasNext())
		{
			Pair pair=conv.next();
			System.out.println(pair.getKey()+": "+pair.getValue());
		}
	}
}
