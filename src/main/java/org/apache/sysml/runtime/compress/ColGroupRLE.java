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
import org.apache.sysml.runtime.compress.utils.ConverterUtils;
import org.apache.sysml.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;


/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends ColGroupBitmap 
{
	private static final long serialVersionUID = 7450232907594748177L;

	public ColGroupRLE() {
		super(CompressionType.RLE_BITMAP);
	}
	
	/**
	 * Main constructor. Constructs and stores the necessary bitmaps.
	 * 
	 * @param colIndices
	 *            indices (within the block) of the columns included in this
	 *            column
	 * @param numRows
	 *            total number of rows in the parent block
	 * @param ubm
	 *            Uncompressed bitmap representation of the block
	 */
	public ColGroupRLE(int[] colIndices, int numRows, UncompressedBitmap ubm) 
	{
		super(CompressionType.RLE_BITMAP, colIndices, numRows, ubm);
		
		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for( int k=0; k<numVals; k++ ) {
			lbitmaps[k] = BitmapEncoder.genRLEBitmap(ubm.getOffsetsList(k));
			totalLen += lbitmaps[k].length;
		}
		
		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);
	}

	/**
	 * Constructor for internal use.
	 */
	public ColGroupRLE(int[] colIndices, int numRows, boolean zeros, double[] values, char[] bitmaps, int[] bitmapOffs) {
		super(CompressionType.RLE_BITMAP, colIndices, numRows, zeros, values);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public Iterator<Integer> getDecodeIterator(int k) {
		return new BitmapDecoderRLE(_data, _ptr[k], len(k)); 
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) 
	{
		if( LOW_LEVEL_OPT && getNumValues() > 1 )
		{
			final int blksz = 128 * 1024;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			
			//position and start offset arrays
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			
			//cache conscious append via horizontal scans 
			for( int bi=rl; bi<ru; bi+=blksz ) {
				int bimax = Math.min(bi+blksz, ru);					
				for (int k=0, off=0; k < numVals; k++, off+=numCols) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					int start = astart[k];
					for( ; bix<blen & start<bimax; bix+=2) {
						start += _data[boff + bix];
						int len = _data[boff + bix+1];
						for( int i=Math.max(rl,start); i<Math.min(start+len,ru); i++ )
							for( int j=0; j<numCols; j++ )
								if( _values[off+j]!=0 )
									target.appendValue(i, _colIndexes[j], _values[off+j]);
						start += len;
					}
					apos[k] = bix;	
					astart[k] = start;
				}
			}
		}
		else
		{
			//call generic decompression with decoder
			super.decompressToBlock(target, rl, ru);
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colixTargets) 
	{
		if( LOW_LEVEL_OPT && getNumValues() > 1 )
		{
			final int blksz = 128 * 1024;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final int n = getNumRows();
			
			//position and start offset arrays
			int[] apos = new int[numVals];
			int[] astart = new int[numVals];
			int[] cix = new int[numCols];
			
			//prepare target col indexes
			for( int j=0; j<numCols; j++ )
				cix[j] = colixTargets[_colIndexes[j]];
			
			//cache conscious append via horizontal scans 
			for( int bi=0; bi<n; bi+=blksz ) {
				int bimax = Math.min(bi+blksz, n);					
				for (int k=0, off=0; k < numVals; k++, off+=numCols) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					if( bix >= blen ) 
						continue;
					int start = astart[k];
					for( ; bix<blen & start<bimax; bix+=2) {
						start += _data[boff + bix];
						int len = _data[boff + bix+1];
						for( int i=start; i<start+len; i++ )
							for( int j=0; j<numCols; j++ )
								if( _values[off+j]!=0 )
									target.appendValue(i, cix[j], _values[off+j]);
						start += len;
					}
					apos[k] = bix;	
					astart[k] = start;
				}
			}
		}
		else
		{
			//call generic decompression with decoder
			super.decompressToBlock(target, colixTargets);
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) 
	{
		if( LOW_LEVEL_OPT && getNumValues() > 1 )
		{
			final int blksz = 128 * 1024;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final int n = getNumRows();
			double[] c = target.getDenseBlock();
			
			//position and start offset arrays
			int[] apos = new int[numVals];
			int[] astart = new int[numVals];
			
			//cache conscious append via horizontal scans 
			for( int bi=0; bi<n; bi+=blksz ) {
				int bimax = Math.min(bi+blksz, n);					
				for (int k=0, off=0; k < numVals; k++, off+=numCols) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					if( bix >= blen ) 
						continue;
					int start = astart[k];
					for( ; bix<blen & start<bimax; bix+=2) {
						start += _data[boff + bix];
						int len = _data[boff + bix+1];
						for( int i=start; i<start+len; i++ )
							c[i] = _values[off+colpos];
						start += len;
					}
					apos[k] = bix;	
					astart[k] = start;
				}
			}
			
			target.recomputeNonZeros();
		}
		else
		{
			//call generic decompression with decoder
			super.decompressToBlock(target, colpos);
		}
	}
	
	@Override
	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru)
			throws DMLRuntimeException 
	{
		double[] b = ConverterUtils.getDenseVector(vector);
		double[] c = result.getDenseBlock();
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		//prepare reduced rhs w/ relevant values
		double[] sb = new double[numCols];
		for (int j = 0; j < numCols; j++) {
			sb[j] = b[_colIndexes[j]];
		}
		
		if( LOW_LEVEL_OPT && numVals > 1 
			&& _numRows > BitmapEncoder.BITMAP_BLOCK_SZ )
		{
			//L3 cache alignment, see comment rightMultByVector OLE column group 
			//core difference of RLE to OLE is that runs are not segment alignment,
			//which requires care of handling runs crossing cache-buckets
			final int blksz = ColGroupBitmap.WRITE_CACHE_BLKSZ; 
			
			//step 1: prepare position and value arrays
			
			//current pos / values per RLE list
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			double[] aval = preaggValues(numVals, sb);
			
			//step 2: cache conscious matrix-vector via horizontal scans 
			for( int bi=rl; bi<ru; bi+=blksz ) 
			{
				int bimax = Math.min(bi+blksz, ru);
					
				//horizontal segment scan, incl pos maintenance
				for (int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					double val = aval[k];
					int bix = apos[k];
					int start = astart[k];
					
					//compute partial results, not aligned
					while( bix<blen ) {
						int lstart = _data[boff + bix];
						int llen = _data[boff + bix + 1];
						LinearAlgebraUtils.vectAdd(val, c, Math.max(bi, start+lstart), 
								Math.min(start+lstart+llen,bimax) - Math.max(bi, start+lstart));
						if(start+lstart+llen >= bimax)
							break;
						start += lstart + llen;
						bix += 2;
					}
					
					apos[k] = bix;	
					astart[k] = start;
				}
			}
		}
		else
		{
			for (int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				double val = sumValues(k, sb);
				int bix = 0;
				int start = 0;
				
				//scan to beginning offset if necessary 
				if( rl > 0 ) { //rl aligned with blksz	
					while( bix<blen ) {	
						int lstart = _data[boff + bix]; //start
						int llen = _data[boff + bix + 1]; //len
						if( start+lstart+llen >= rl )
							break;
						start += lstart + llen;
						bix += 2;
					}
				}
				
				//compute partial results, not aligned
				while( bix<blen ) {
					int lstart = _data[boff + bix];
					int llen = _data[boff + bix + 1];
					LinearAlgebraUtils.vectAdd(val, c, Math.max(rl, start+lstart), 
							Math.min(start+lstart+llen,ru) - Math.max(rl, start+lstart));
					if(start+lstart+llen >= ru)
						break;
					start += lstart + llen;
					bix += 2;
				}
			}
		}
	}

	@Override
	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result)
			throws DMLRuntimeException 
	{		
		double[] a = ConverterUtils.getDenseVector(vector);
		double[] c = result.getDenseBlock();
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final int n = getNumRows();
		
		if( LOW_LEVEL_OPT && numVals > 1 
			&& _numRows > BitmapEncoder.BITMAP_BLOCK_SZ ) 
		{
			final int blksz = ColGroupBitmap.READ_CACHE_BLKSZ; 
			
			//step 1: prepare position and value arrays
			
			//current pos per OLs / output values
			int[] apos = new int[numVals];
			int[] astart = new int[numVals];
			double[] cvals = new double[numVals];
			
			//step 2: cache conscious matrix-vector via horizontal scans 
			for( int ai=0; ai<n; ai+=blksz ) 
			{
				int aimax = Math.min(ai+blksz, n);
				
				//horizontal scan, incl pos maintenance
				for (int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);						
					int bix = apos[k];
					int start = astart[k];
					
					//compute partial results, not aligned
					while( bix<blen & start<aimax ) {
						start += _data[boff + bix];
						int len = _data[boff + bix+1];
						cvals[k] += LinearAlgebraUtils.vectSum(a, start, len);
						start += len;
						bix += 2;
					}
					
					apos[k] = bix;	
					astart[k] = start;
				}
			}
			
			//step 3: scale partial results by values and write to global output
			for (int k = 0, valOff=0; k < numVals; k++, valOff+=numCols)
				for( int j = 0; j < numCols; j++ )
					c[ _colIndexes[j] ] += cvals[k] * _values[valOff+j];
			
		}
		else
		{
			//iterate over all values and their bitmaps
			for (int k=0, valOff=0; k<numVals; k++, valOff+=numCols) 
			{	
				int boff = _ptr[k];
				int blen = len(k);
				
				double vsum = 0;
				int curRunEnd = 0;
				for ( int bix = 0; bix < blen; bix+=2 ) {
					int curRunStartOff = curRunEnd + _data[boff+bix];
					int curRunLen = _data[boff+bix+1];
					vsum += LinearAlgebraUtils.vectSum(a, curRunStartOff, curRunLen);
					curRunEnd = curRunStartOff + curRunLen;
				}
				
				//scale partial results by values and write results
				for( int j = 0; j < numCols; j++ )
					c[ _colIndexes[j] ] += vsum * _values[ valOff+j ];
			}
		}
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op)
			throws DMLRuntimeException 
	{
		double val0 = op.executeScalar(0);
		
		//fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if( op.sparseSafe || val0==0 ) {
			return new ColGroupRLE(_colIndexes, _numRows, _zeros,
					applyScalarOp(op), _data, _ptr);
		}
		
		//slow path: sparse-unsafe operations (potentially create new bitmap)
		//note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if( loff.length==0 ) { //empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, true,
					applyScalarOp(op), _data, _ptr);
		}
		
		double[] rvalues = applyScalarOp(op, val0, getNumCols());		
		char[] lbitmap = BitmapEncoder.genRLEBitmap(loff);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length+lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length+1);
		rbitmapOffs[rbitmapOffs.length-1] = rbitmaps.length; 
		
		return new ColGroupRLE(_colIndexes, _numRows, loff.length<_numRows,
				rvalues, rbitmaps, rbitmapOffs);
	}
	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) 
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
				computeRowSums(result, kplus);
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
				computeMxx(result, builtin);
			else if( op.indexFn instanceof ReduceCol )
				computeRowMxx(result, builtin);
			else if( op.indexFn instanceof ReduceRow )
				computeColMxx(result, builtin);
		}
	}
	
	/**
	 * 
	 * @param result
	 */
	private void computeSum(MatrixBlock result, KahanFunction kplus)
	{
		KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
		
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int valOff = k * numCols;
			int curRunEnd = 0;
			int count = 0;
			for (int bix = 0; bix < blen; bix+=2) {
				int curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix+1];
				count += curRunEnd-curRunStartOff;
			}
			
			//scale counts by all values
			for( int j = 0; j < numCols; j++ )
				kplus.execute3(kbuff, _values[ valOff+j ], count);
		}
		
		result.quickSetValue(0, 0, kbuff._sum);
		result.quickSetValue(0, 1, kbuff._correction);
	}
	
	/**
	 * 
	 * @param result
	 */
	private void computeRowSums(MatrixBlock result, KahanFunction kplus)
	{
		KahanObject kbuff = new KahanObject(0, 0);
		final int numVals = getNumValues();
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			double val = sumValues(k);
					
			if (val != 0.0) {
				int curRunStartOff = 0;
				int curRunEnd = 0;
				for (int bix = 0; bix < blen; bix+=2) {
					curRunStartOff = curRunEnd + _data[boff+bix];
					curRunEnd = curRunStartOff + _data[boff+bix+1];
					for (int rix = curRunStartOff; rix < curRunEnd; rix++) {
						kbuff.set(result.quickGetValue(rix, 0), result.quickGetValue(rix, 1));
						kplus.execute2(kbuff, val);
						result.quickSetValue(rix, 0, kbuff._sum);
						result.quickSetValue(rix, 1, kbuff._correction);
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param result
	 */
	private void computeColSums(MatrixBlock result, KahanFunction kplus)
	{
		KahanObject kbuff = new KahanObject(0, 0);
		
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int valOff = k * numCols;
			int curRunEnd = 0;
			int count = 0;
			for (int bix=0; bix < blen; bix+=2) {
				int curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix+1];
				count += curRunEnd-curRunStartOff;
			}
			
			//scale counts by all values
			for( int j = 0; j < numCols; j++ ) {
				kbuff.set(result.quickGetValue(0, _colIndexes[j]),result.quickGetValue(1, _colIndexes[j]));
				kplus.execute3(kbuff, _values[ valOff+j ], count);
				result.quickSetValue(0, _colIndexes[j], kbuff._sum);
				result.quickSetValue(1, _colIndexes[j], kbuff._correction);
			}
		}
	}
	

	/**
	 * 
	 * @param result
	 */
	private void computeRowMxx(MatrixBlock result, Builtin builtin)
	{
		//NOTE: zeros handled once for all column groups outside
		
		final int numVals = getNumValues();
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			double val = mxxValues(k, builtin);
					
			if (val != 0.0) {
				int curRunStartOff = 0;
				int curRunEnd = 0;
				for (int bix = 0; bix < blen; bix+=2) {
					curRunStartOff = curRunEnd + _data[boff+bix];
					curRunEnd = curRunStartOff + _data[boff+bix+1];
					for (int rix = curRunStartOff; rix < curRunEnd; rix++) {
						result.quickSetValue(rix, 0, 
								builtin.execute2(result.quickGetValue(rix, 0), val));
					}
				}
			}
		}
	}
	
	public boolean[] computeZeroIndicatorVector()
		throws DMLRuntimeException 
	{	
		boolean[] ret = new boolean[_numRows];
		final int numVals = getNumValues();

		//initialize everything with zero
		Arrays.fill(ret, true);
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			
			int curRunStartOff = 0;
			int curRunEnd = 0;
			for (int bix = 0; bix < blen; bix+=2) {
				curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix + 1];
				Arrays.fill(ret, curRunStartOff, curRunEnd, false);
			}
		}
		
		return ret;
	}
	
	@Override
	protected void countNonZerosPerRow(int[] rnnz, int rl, int ru)
	{
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		//current pos / values per RLE list
		int[] astart = new int[numVals];
		int[] apos = skipScan(numVals, rl, astart);
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = apos[k];
					
			int curRunStartOff = 0;
			int curRunEnd = 0;
			for( ; bix < blen && curRunStartOff<ru; bix+=2) {
				curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix + 1];
				for( int i=Math.max(curRunStartOff,rl); i<Math.min(curRunEnd, ru); i++ )
					rnnz[i-rl] += numCols;
			}
		}
	}
	
	/////////////////////////////////
	// internal helper functions

	
	/**
	 * Scans to given row_lower position by scanning run length 
	 * fields. Returns array of positions for all values and modifies
	 * given array of start positions for all values too. 
	 * 
	 * @param numVals
	 * @param rl
	 * @param astart
	 * @return
	 */
	private int[] skipScan(int numVals, int rl, int[] astart) {
		int[] apos = new int[numVals]; 
		
		if( rl > 0 ) { //rl aligned with blksz	
			for (int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = 0;
				int start = 0;
				while( bix<blen ) {	
					int lstart = _data[boff + bix]; //start
					int llen = _data[boff + bix + 1]; //len
					if( start+lstart+llen >= rl )
						break;
					start += lstart + llen;
					bix += 2;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
		
		return apos;
	}
}
