/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysds.runtime.matrix.data;


import java.util.Iterator;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.util.UtilFunctions;



public class BinaryBlockToTextCellConverter implements 
Converter<MatrixIndexes, MatrixBlock, NullWritable, Text>
{	
	private Iterator<IJV> sparseIterator=null;
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
	private Pair<NullWritable, Text> pair=new Pair<>(NullWritable.get(), value);
	
	private void reset()
	{
		sparseIterator=null;
		denseArray=null;
		denseArraySize=0;
		nextInDenseArray=-1;
		sparse=true;
		thisBlockWidth=0;
	}
	
	/**
	 * Before calling convert, please make sure to setBlockSize(blen, blen);
	 */
	@Override
	public void convert(MatrixIndexes k1, MatrixBlock v1) {
		reset();
		startIndexes.setIndexes(UtilFunctions.computeCellIndex(k1.getRowIndex(), brow,0), 
				UtilFunctions.computeCellIndex(k1.getColumnIndex(),bcolumn,0));
		sparse=v1.isInSparseFormat();
		thisBlockWidth=v1.getNumColumns();
		if(sparse)
		{
			sparseIterator=v1.getSparseBlockIterator();
		}
		else
		{
			if(v1.getDenseBlock()==null)
				return;
			if(v1.getDenseBlock() instanceof DenseBlockFP64DEDUP){
				DenseBlockFP64DEDUP db = (DenseBlockFP64DEDUP) v1.getDenseBlock();
				denseArray = new double[v1.rlen*v1.clen];
				for (int i = 0; i < v1.rlen; i++) {
					double[] row = db.values(i);
					for (int j = 0; j < v1.clen; j++) {
						denseArray[i*v1.clen + j] = row[j];
					}
				}
			} else
				denseArray=v1.getDenseBlockValues();
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
			IJV cell = sparseIterator.next();
			i = cell.getI() + startIndexes.getRowIndex();
			j = cell.getJ() + startIndexes.getColumnIndex();
			v = cell.getV();
		}
		else {
			if(denseArray==null)
				return null;
			i=startIndexes.getRowIndex() + nextInDenseArray/thisBlockWidth;
			j=startIndexes.getColumnIndex() + nextInDenseArray%thisBlockWidth;
			v=denseArray[nextInDenseArray];
			nextInDenseArray++;
		}
		value.set(i+" "+j+" "+v);
		return pair;
	}

	@Override
	public void setBlockSize(int nr, int nc) {
		brow=nr;
		bcolumn=nc;
	}
}
