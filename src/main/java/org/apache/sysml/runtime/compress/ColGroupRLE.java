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
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ReduceAll;
import org.apache.sysml.runtime.functionobjects.ReduceCol;
import org.apache.sysml.runtime.functionobjects.ReduceRow;
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
		for( int i=0; i<numVals; i++ ) {
			lbitmaps[i] = BitmapEncoder.genRLEBitmap(ubm.getOffsetsList(i));
			totalLen += lbitmaps[i].length;
		}
		
		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);
	}

	/**
	 * Constructor for internal use.
	 */
	public ColGroupRLE(int[] colIndices, int numRows, double[] values, char[] bitmaps, int[] bitmapOffs) {
		super(CompressionType.RLE_BITMAP, colIndices, numRows, values);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public Iterator<Integer> getDecodeIterator(int bmpIx) {
		return new BitmapDecoderRLE(_data, _ptr[bmpIx], len(bmpIx)); 
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target) 
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
			
			//cache conscious append via horizontal scans 
			for( int bi=0; bi<n; bi+=blksz ) {
				int bimax = Math.min(bi+blksz, n);					
				for (int k=0, off=0; k < numVals; k++, off+=numCols) {
					int bitmapOff = _ptr[k];
					int bitmapLen = len(k);
					int bufIx = apos[k];
					int start = astart[k];
					for( ; bufIx<bitmapLen & start<bimax; bufIx+=2) {
						start += _data[bitmapOff + bufIx];
						int len = _data[bitmapOff + bufIx+1];
						for( int i=start; i<start+len; i++ )
							for( int j=0; j<numCols; j++ )
								if( _values[off+j]!=0 )
									target.appendValue(i, _colIndexes[j], _values[off+j]);
						start += len;
					}
					apos[k] = bufIx;	
					astart[k] = start;
				}
			}
		}
		else
		{
			//call generic decompression with decoder
			super.decompressToBlock(target);
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
					int bitmapOff = _ptr[k];
					int bitmapLen = len(k);
					int bufIx = apos[k];
					if( bufIx >= bitmapLen ) 
						continue;
					int start = astart[k];
					for( ; bufIx<bitmapLen & start<bimax; bufIx+=2) {
						start += _data[bitmapOff + bufIx];
						int len = _data[bitmapOff + bufIx+1];
						for( int i=start; i<start+len; i++ )
							for( int j=0; j<numCols; j++ )
								if( _values[off+j]!=0 )
									target.appendValue(i, cix[j], _values[off+j]);
						start += len;
					}
					apos[k] = bufIx;	
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
					int bitmapOff = _ptr[k];
					int bitmapLen = len(k);
					int bufIx = apos[k];
					if( bufIx >= bitmapLen ) 
						continue;
					int start = astart[k];
					for( ; bufIx<bitmapLen & start<bimax; bufIx+=2) {
						start += _data[bitmapOff + bufIx];
						int len = _data[bitmapOff + bufIx+1];
						for( int i=start; i<start+len; i++ )
							c[i] = _values[off+colpos];
						start += len;
					}
					apos[k] = bufIx;	
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
			int[] apos = new int[numVals];
			int[] astart = new int[numVals];
			double[] aval = new double[numVals];
			
			//skip-scan to beginning for all OLs 
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
			
			//pre-aggregate values per OLs
			for( int k = 0; k < numVals; k++ )
				aval[k] = sumValues(k, sb);
					
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
					int bitmapOff = _ptr[k];
					int bitmapLen = len(k);						
					int bufIx = apos[k];
					int start = astart[k];
					
					//compute partial results, not aligned
					while( bufIx<bitmapLen & start<aimax ) {
						start += _data[bitmapOff + bufIx];
						int len = _data[bitmapOff + bufIx+1];
						cvals[k] += LinearAlgebraUtils.vectSum(a, start, len);
						start += len;
						bufIx += 2;
					}
					
					apos[k] = bufIx;	
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
			for (int bitmapIx=0, valOff=0; bitmapIx<numVals; bitmapIx++, valOff+=numCols) 
			{	
				int bitmapOff = _ptr[bitmapIx];
				int bitmapLen = len(bitmapIx);
				
				double vsum = 0;
				int curRunEnd = 0;
				for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
					int curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
					int curRunLen = _data[bitmapOff+bufIx + 1];
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
			return new ColGroupRLE(_colIndexes, _numRows, 
					applyScalarOp(op), _data, _ptr);
		}
		
		//slow path: sparse-unsafe operations (potentially create new bitmap)
		//note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if( loff.length==0 ) { //empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, 
					applyScalarOp(op), _data, _ptr);
		}
		
		double[] rvalues = applyScalarOp(op, val0, getNumCols());		
		char[] lbitmap = BitmapEncoder.genRLEBitmap(loff);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length+lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length+1);
		rbitmapOffs[rbitmapOffs.length-1] = rbitmaps.length; 
		
		return new ColGroupRLE(_colIndexes, _numRows, 
				rvalues, rbitmaps, rbitmapOffs);
	}
	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result) 
		throws DMLRuntimeException 
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
	
	/**
	 * 
	 * @param result
	 */
	private void computeSum(MatrixBlock result, KahanFunction kplus)
	{
		KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
		
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		for (int bitmapIx = 0; bitmapIx < numVals; bitmapIx++) {
			int bitmapOff = _ptr[bitmapIx];
			int bitmapLen = len(bitmapIx);
			int valOff = bitmapIx * numCols;
			int curRunEnd = 0;
			int count = 0;
			for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
				int curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
				curRunEnd = curRunStartOff + _data[bitmapOff+bufIx + 1];
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
		
		for (int bitmapIx = 0; bitmapIx < numVals; bitmapIx++) {
			int bitmapOff = _ptr[bitmapIx];
			int bitmapLen = len(bitmapIx);
			double val = sumValues(bitmapIx);
					
			if (val != 0.0) {
				int curRunStartOff = 0;
				int curRunEnd = 0;
				for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
					curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
					curRunEnd = curRunStartOff + _data[bitmapOff+bufIx + 1];
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
		
		for (int bitmapIx = 0; bitmapIx < numVals; bitmapIx++) {
			int bitmapOff = _ptr[bitmapIx];
			int bitmapLen = len(bitmapIx);
			int valOff = bitmapIx * numCols;
			int curRunEnd = 0;
			int count = 0;
			for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
				int curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
				curRunEnd = curRunStartOff + _data[bitmapOff+bufIx + 1];
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
	
	public boolean[] computeZeroIndicatorVector()
		throws DMLRuntimeException 
	{	
		boolean[] ret = new boolean[_numRows];
		final int numVals = getNumValues();

		//initialize everything with zero
		Arrays.fill(ret, true);
		
		for (int bitmapIx = 0; bitmapIx < numVals; bitmapIx++) {
			int bitmapOff = _ptr[bitmapIx];
			int bitmapLen = len(bitmapIx);
			
			int curRunStartOff = 0;
			int curRunEnd = 0;
			for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
				curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
				curRunEnd = curRunStartOff + _data[bitmapOff+bufIx + 1];
				Arrays.fill(ret, curRunStartOff, curRunEnd, false);
			}
		}
		
		return ret;
	}
	
	@Override
	protected void countNonZerosPerRow(int[] rnnz)
	{
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		for (int k = 0; k < numVals; k++) {
			int bitmapOff = _ptr[k];
			int bitmapLen = len(k);
			
			int curRunStartOff = 0;
			int curRunEnd = 0;
			for (int bufIx = 0; bufIx < bitmapLen; bufIx += 2) {
				curRunStartOff = curRunEnd + _data[bitmapOff+bufIx];
				curRunEnd = curRunStartOff + _data[bitmapOff+bufIx + 1];
				for( int i=curRunStartOff; i<curRunEnd; i++ )
					rnnz[i] += numCols;
			}
		}
	}
}
