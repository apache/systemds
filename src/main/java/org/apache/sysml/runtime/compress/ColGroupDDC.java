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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * Class to encapsulate information about a column group that is encoded with
 * dense dictionary encoding (DDC).
 * 
 * NOTE: zero values are included at position 0 in the value dictionary, which
 * simplifies various operations such as counting the number of non-zeros.
 */
public abstract class ColGroupDDC extends ColGroupValue 
{
	private static final long serialVersionUID = -3204391646123465004L;

	public ColGroupDDC() {
		super();
	}
	
	public ColGroupDDC(int[] colIndices, int numRows, UncompressedBitmap ubm) {
		super(colIndices, numRows, ubm);
	}
	
	protected ColGroupDDC(int[] colIndices, int numRows, double[] values) {
		super(colIndices, numRows, values);
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		for( int i = rl; i < ru; i++ ) {
			for( int colIx = 0; colIx < _colIndexes.length; colIx++ ) {
				int col = _colIndexes[colIx];
				double cellVal = getData(i, colIx);
				target.quickSetValue(i, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		for( int i = 0; i < nrow; i++ ) {
			for( int colIx = 0; colIx < ncol; colIx++ ) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = getData(i, colIx);
				target.quickSetValue(i, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		int nrow = getNumRows();
		for( int i = 0; i < nrow; i++ ) {
			double cellVal = getData(i, colpos);
			target.quickSetValue(i, 0, cellVal);
		}
	}
	
	@Override
	public double get(int r, int c) {
		//find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if( ix < 0 )
			throw new RuntimeException("Column index "+c+" not in DDC group.");
		
		//get value
		return getData(r, ix);
	}
	

	@Override
	protected void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		int ncol = getNumCols();
		for( int i = rl; i < ru; i++ ) {
			int lnnz = 0;
			for( int colIx=0; colIx < ncol; colIx++ )
				lnnz += (getData(i, colIx) != 0) ? 1 : 0;
			rnnz[i-rl] += lnnz;
		}
	}
	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result, int rl, int ru)
		throws DMLRuntimeException 
	{
		//sum and sumsq (reduceall/reducerow over tuples and counts)
		if( op.aggOp.increOp.fn instanceof KahanPlus || op.aggOp.increOp.fn instanceof KahanPlusSq ) 
		{
			KahanFunction kplus = (op.aggOp.increOp.fn instanceof KahanPlus) ?
					KahanPlus.getKahanPlusFnObject() : KahanPlusSq.getKahanPlusSqFnObject();
			
			if( op.indexFn instanceof ReduceAll )
				computeSum(result, kplus);
			else if( op.indexFn instanceof ReduceCol )
				computeRowSums(result, kplus, rl, ru);
			else if( op.indexFn instanceof ReduceRow )
				computeColSums(result, kplus);
		}
		//min and max (reduceall/reducerow over tuples only)
		else if(op.aggOp.increOp.fn instanceof Builtin 
				&& (((Builtin)op.aggOp.increOp.fn).getBuiltinCode()==BuiltinCode.MAX 
				|| ((Builtin)op.aggOp.increOp.fn).getBuiltinCode()==BuiltinCode.MIN)) 
		{		
			Builtin builtin = (Builtin) op.aggOp.increOp.fn;

			if( op.indexFn instanceof ReduceAll )
				computeMxx(result, builtin, false);
			else if( op.indexFn instanceof ReduceCol )
				computeRowMxx(result, builtin, rl, ru);
			else if( op.indexFn instanceof ReduceRow )
				computeColMxx(result, builtin, false);
		}
	}
	
	protected void computeSum(MatrixBlock result, KahanFunction kplus) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
		
		for( int i=0; i<nrow; i++ )
			for( int j=0; j<ncol; j++ )
				kplus.execute2(kbuff, getData(i, j));
		
		result.quickSetValue(0, 0, kbuff._sum);
		result.quickSetValue(0, 1, kbuff._correction);
	}
	
	protected void computeColSums(MatrixBlock result, KahanFunction kplus) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		KahanObject[] kbuff = new KahanObject[getNumCols()];
		for( int j=0; j<ncol; j++ )
			kbuff[j] = new KahanObject(result.quickGetValue(0, _colIndexes[j]), 
					result.quickGetValue(1, _colIndexes[j]));
		
		for( int i=0; i<nrow; i++ )
			for( int j=0; j<ncol; j++ )
				kplus.execute2(kbuff[j], getData(i, j));
		
		for( int j=0; j<ncol; j++ ) {
			result.quickSetValue(0, _colIndexes[j], kbuff[j]._sum);
			result.quickSetValue(1, _colIndexes[j], kbuff[j]._correction);
		}
	}

	protected void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru) {
		int ncol = getNumCols();
		KahanObject kbuff = new KahanObject(0, 0);
		
		for( int i=rl; i<ru; i++ ) {
			kbuff.set(result.quickGetValue(i, 0), result.quickGetValue(i, 1));
			for( int j=0; j<ncol; j++ )
				kplus.execute2(kbuff, getData(i, j));
			result.quickSetValue(i, 0, kbuff._sum);
			result.quickSetValue(i, 1, kbuff._correction);
		}
	}
	
	protected void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru) {
		double[] c = result.getDenseBlock();
		int ncol = getNumCols();
		
		for( int i=rl; i<ru; i++ )
			for( int j=0; j<ncol; j++ )
				c[i] = builtin.execute2(c[i], getData(i, j));
	}
	
	protected final void postScaling(double[] vals, double[] c) {
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		
		for( int k=0, valOff=0; k<numVals; k++, valOff+=ncol ) {
			double aval = vals[k];
			for( int j=0; j<ncol; j++ ) {
				int colIx = _colIndexes[j];
				c[colIx] += aval * _values[valOff+j];
			}	
		}
	}

	/**
	 * Generic get value for byte-length-agnostic access.
	 * 
	 * @param r global row index
	 * @param colIx local column index 
	 * @return value
	 */
	protected abstract double getData(int r, int colIx);
	
	/**
	 * Generic set value for byte-length-agnostic write 
	 * of encoded value.
	 * 
	 * @param r global row index
	 * @param code encoded value 
	 */
	protected abstract void setData(int r, int code);
	
	@Override
	public long estimateInMemorySize() {
		return super.estimateInMemorySize();
	}
	
	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		//DDC iterator is always row major, so no need for custom handling
		return new DDCIterator(rl, ru, inclZeros);
	}
	
	private class DDCIterator implements Iterator<IJV>
	{
		//iterator configuration 
		private final int _ru;
		private final boolean _inclZeros;
		
		//iterator state
		private final IJV _buff = new IJV(); 
		private int _rpos = -1;
		private int _cpos = -1;
		private double _value = 0;
		
		public DDCIterator(int rl, int ru, boolean inclZeros) {
			_ru = ru;
			_inclZeros = inclZeros;
			_rpos = rl;
			_cpos = -1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}

		@Override
		public IJV next() {
			_buff.set(_rpos, _colIndexes[_cpos], _value);
			getNextValue();
			return _buff;
		}
		
		private void getNextValue() {
			do {
				boolean nextRow = (_cpos+1 >= getNumCols());
				_rpos += nextRow ? 1 : 0; 
				_cpos = nextRow ? 0 : _cpos+1;
				if( _rpos >= _ru )
					return; //reached end
				_value = getData(_rpos, _cpos);
			}
			while( !_inclZeros && _value==0);
		}
	}
}
