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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.utils.ConverterUtils;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with
 * dense dictionary encoding (DDC) using 2 byte codes.
 */
public class ColGroupDDC2 extends ColGroupDDC 
{
	private static final long serialVersionUID = -3995768285207071013L;
	
	private static final int MAX_TMP_VALS = 32*1024;
	
	private char[] _data;

	public ColGroupDDC2() {
		super();
	}
	
	public ColGroupDDC2(int[] colIndices, int numRows, UncompressedBitmap ubm) {
		super(colIndices, numRows, ubm);
		_data = new char[numRows];
		
		int numVals = ubm.getNumValues();
		int numCols = ubm.getNumColumns();
		
		//materialize zero values, if necessary
		if( ubm.getNumOffsets() < (long)numRows * numCols ) {
			int zeroIx = containsAllZeroValue();
			if( zeroIx < 0 ) {
				zeroIx = numVals;
				_values = Arrays.copyOf(_values, _values.length+numCols);
			}
			Arrays.fill(_data, (char)zeroIx);
		}
		
		//iterate over values and write dictionary codes
		for( int i=0; i<numVals; i++ ) {
			int[] tmpList = ubm.getOffsetsList(i).extractValues();
			int tmpListSize = ubm.getNumOffsets(i); 
			for( int k=0; k<tmpListSize; k++ )
				_data[tmpList[k]] = (char)i;
		}
	}
	
	public ColGroupDDC2(int[] colIndices, int numRows, double[] values, char[] data) {
		super(colIndices, numRows, values);
		_data = data;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.DDC2;
	}
	
	@Override
	protected double getData(int r, int colIx) {
		return _values[_data[r]*getNumCols()+colIx];
	}

	@Override
	protected void setData(int r, int code) {
		_data[r] = (char)code;
	}
	
	@Override
	public void write(DataOutput out) throws IOException {
		int numCols = getNumCols();
		int numVals = getNumValues();
		out.writeInt(_numRows);
		out.writeInt(numCols);
		out.writeInt(numVals);
		
		//write col indices
		for( int i=0; i<_colIndexes.length; i++ )
			out.writeInt( _colIndexes[i] );
		
		//write distinct values
		for( int i=0; i<_values.length; i++ )
			out.writeDouble(_values[i]);

		//write data
		for( int i=0; i<_numRows; i++ )
			out.writeChar(_data[i]);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		_numRows = in.readInt();
		int numCols = in.readInt();
		int numVals = in.readInt();
		
		//read col indices
		_colIndexes = new int[ numCols ];
		for( int i=0; i<numCols; i++ )
			_colIndexes[i] = in.readInt();
		
		//read distinct values
		_values = new double[numVals*numCols];
		for( int i=0; i<numVals*numCols; i++ )
			_values[i] = in.readDouble();
		
		//read data
		_data = new char[_numRows];
		for( int i=0; i<_numRows; i++ )
			_data[i] = in.readChar();
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = 12; //header
		//col indices
		ret += 4 * _colIndexes.length; 
		//distinct values (groups of values)
		ret += 8 * _values.length;
		//data
		ret += 2 * _data.length;
		
		return ret;
	}
	
	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		
		//adding data size
		if (_data != null)
			size += 2 * _data.length;
	
		return size;
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		int ncol = getNumCols();
		for( int i = rl; i < ru; i++ )
			for( int j=0; j<ncol; j++ )
				target.appendValue(i, _colIndexes[j], _values[_data[i]*ncol+j]);
		//note: append ok because final sort per row 
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		int nrow = getNumRows();
		int ncol = getNumCols();
		double[] c = target.getDenseBlock();
		int nnz = 0;
		for( int i = 0; i < nrow; i++ )
			nnz += ((c[i] = _values[_data[i]*ncol+colpos])!=0) ? 1 : 0;
		target.setNonZeros(nnz);
	}
	
	@Override 
	public int[] getCounts() {
		final int nrow = getNumRows();
		final int numVals = getNumValues();
		
		int[] counts = new int[numVals];
		for( int i=0; i<nrow; i++ ) {
			counts[_data[i]] ++;
		}
		
		return counts;
	}
	
	@Override
	protected void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		
		//pre-aggregate nnz per value tuple
		int[] counts = new int[numVals];
		for( int k=0, valOff=0; k<numVals; k++, valOff+=ncol )
			for( int j=0; j<ncol; j++ )
				counts[k] += (_values[valOff+j]!=0) ? 1 : 0;
		
		//scan data and add counts to output rows
		for( int i = rl; i < ru; i++ )
			rnnz[i-rl] += counts[_data[i]];
	}
	
	@Override
	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru) throws DMLRuntimeException {
		double[] b = ConverterUtils.getDenseVector(vector);
		double[] c = result.getDenseBlock();
		final int numCols = getNumCols();
		final int numVals = getNumValues();

		//prepare reduced rhs w/ relevant values
		double[] sb = new double[numCols];
		for (int j = 0; j < numCols; j++) {
			sb[j] = b[_colIndexes[j]];
		}
		
		//pre-aggregate all distinct values 
		double[] vals = preaggValues(numVals, sb);

		//iterative over codes and add to output
		for( int i=rl; i<ru; i++ )
			c[i] += vals[_data[i]];
	}

	@Override
	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result) 
		throws DMLRuntimeException 
	{
		double[] a = ConverterUtils.getDenseVector(vector);
		double[] c = result.getDenseBlock();
		final int nrow = getNumRows();
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		
		if( 8*numVals < getNumRows() )
		{
			//iterative over codes and pre-aggregate inputs per code
			//temporary array also avoids false sharing in multi-threaded environments
			double[] vals = allocDVector(numVals, true);
			for( int i=0; i<nrow; i++ ) {
				vals[_data[i]] += a[i];
			}
			
			//post-scaling of pre-aggregate with distinct values
			postScaling(vals, c);
		}
		else //general case
		{	
			//iterate over codes, compute all, and add to the result
			for( int i=0; i<nrow; i++ ) {
				double aval = a[i];
				if( aval != 0 )
					for( int j=0, valOff=_data[i]*ncol; j<ncol; j++ )
						c[_colIndexes[j]] += aval * _values[valOff+j];
			}
		}
	}
	
	@Override
	public void leftMultByRowVector(ColGroupDDC a, MatrixBlock result) 
		throws DMLRuntimeException 
	{
		double[] c = result.getDenseBlock();
		final int nrow = getNumRows();
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		
		if( 8*numVals < getNumRows() )
		{
			//iterative over codes and pre-aggregate inputs per code
			//temporary array also avoids false sharing in multi-threaded environments
			double[] vals = allocDVector(numVals, true);
			for( int i=0; i<nrow; i++ ) {
				vals[_data[i]] += a.getData(i, 0);
			}
			
			//post-scaling of pre-aggregate with distinct values
			postScaling(vals, c);
		}
		else //general case
		{	
			//iterate over codes, compute all, and add to the result
			for( int i=0; i<nrow; i++ ) {
				double aval = a.getData(i, 0);
				if( aval != 0 )
					for( int j=0, valOff=_data[i]*ncol; j<ncol; j++ )
						c[_colIndexes[j]] += aval * _values[valOff+j];
			}
		}
	}
	
	@Override
	protected void computeSum(MatrixBlock result, KahanFunction kplus) {
		final int ncol = getNumCols();
		final int numVals = getNumValues();
		
		if( numVals < MAX_TMP_VALS )
		{
			//iterative over codes and count per code
			int[] counts = getCounts();
			
			//post-scaling of pre-aggregate with distinct values
			KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
			for( int k=0, valOff=0; k<numVals; k++, valOff+=ncol ) {
				int cntk = counts[k];
				for( int j=0; j<ncol; j++ )
					kplus.execute3(kbuff, _values[ valOff+j], cntk);
			}
			
			result.quickSetValue(0, 0, kbuff._sum);
			result.quickSetValue(0, 1, kbuff._correction);
		}
		else //general case 
		{
			super.computeSum(result, kplus);
		}
	}
	
	
	@Override
	protected void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru) {
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
		double[] c = result.getDenseBlock();
		
		//pre-aggregate nnz per value tuple
		double[] vals = sumAllValues(kplus, kbuff, false);
		
		//scan data and add to result (use kahan plus not general KahanFunction
		//for correctness in case of sqk+)
		for( int i=rl; i<ru; i++ ) {
			kbuff.set(c[2*i], c[2*i+1]);
			kplus2.execute2(kbuff, vals[_data[i]]);
			c[2*i] = kbuff._sum;
			c[2*i+1] = kbuff._correction;
		}
	}
	
	@Override
	public ColGroup scalarOperation(ScalarOperator op) throws DMLRuntimeException {
		//fast path: sparse-safe and -unsafe operations
		//as zero are represented, it is sufficient to simply apply the scalar op
		return new ColGroupDDC2(_colIndexes, _numRows, applyScalarOp(op), _data);
	}
}
