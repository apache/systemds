/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.runtime.util.UtilFunctions;



public class BinaryBlockToBinaryCellConverter implements 
Converter<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixCell>
{
	
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
	
	private MatrixIndexes returnIndexes=new MatrixIndexes();
	private MatrixCell cell=new MatrixCell();
	private Pair<MatrixIndexes, MatrixCell> pair=new Pair<MatrixIndexes, MatrixCell>(returnIndexes, cell);
	
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
	public Pair<MatrixIndexes, MatrixCell> next() {
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
				IJV tmpcell = sparseIterator.next();
				i = tmpcell.i + startIndexes.getRowIndex();
				j = tmpcell.j + startIndexes.getColumnIndex();
				v = tmpcell.v;
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
		returnIndexes.setIndexes(i,j);
		cell.setValue(v);
		return pair;
	}

	public void setBlockSize(int nr, int nc) {
		brow=nr;
		bcolumn=nc;
	}
}
