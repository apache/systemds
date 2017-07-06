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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.utils.ConverterUtils;
import org.apache.sysml.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;


/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends ColGroupOffset 
{
	private static final long serialVersionUID = 7450232907594748177L;

	private static final Log LOG = LogFactory.getLog(ColGroupRLE.class.getName());
	
	public ColGroupRLE() {
		super();
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
		super(colIndices, numRows, ubm);
		
		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for( int k=0; k<numVals; k++ ) {
			lbitmaps[k] = BitmapEncoder.genRLEBitmap(
				ubm.getOffsetsList(k).extractValues(), ubm.getNumOffsets(k));
			totalLen += lbitmaps[k].length;
		}
		
		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);
		
		//debug output
		double ucSize = MatrixBlock.estimateSizeDenseInMemory(numRows, colIndices.length);
		if( estimateInMemorySize() > ucSize )
			LOG.warn("RLE group larger than UC dense: "+estimateInMemorySize()+" "+ucSize);
	}

	public ColGroupRLE(int[] colIndices, int numRows, boolean zeros, double[] values, char[] bitmaps, int[] bitmapOffs) {
		super(colIndices, numRows, zeros, values);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}
	
	@Override
	public CompressionType getCompType() {
		return CompressionType.RLE_BITMAP;
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
		final int blksz = 128 * 1024;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final int n = getNumRows();
		double[] c = target.getDenseBlock();
		
		//position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = allocIVector(numVals, true);
		
		//cache conscious append via horizontal scans 
		int nnz = 0;
		for( int bi=0; bi<n; bi+=blksz ) {
			int bimax = Math.min(bi+blksz, n);
			Arrays.fill(c, bi, bimax, 0);
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
					Arrays.fill(c, start, start+len, _values[off+colpos]);
					nnz += len;
					start += len;
				}
				apos[k] = bix;	
				astart[k] = start;
			}
		}
		target.setNonZeros(nnz);
	}
	
	@Override 
	public int[] getCounts() {
		final int numVals = getNumValues();
		
		int[] counts = new int[numVals];
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int curRunEnd = 0;
			int count = 0;
			for (int bix = 0; bix < blen; bix+=2) {
				int curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix+1];
				count += curRunEnd-curRunStartOff;
			}
			counts[k] = count;
		}
		
		return counts;
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
			final int blksz = ColGroupOffset.WRITE_CACHE_BLKSZ; 
			
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
			final int blksz = ColGroupOffset.READ_CACHE_BLKSZ; 
			
			//step 1: prepare position and value arrays
			
			//current pos per OLs / output values
			int[] astart = new int[numVals];
			int[] apos = allocIVector(numVals, true);
			double[] cvals = allocDVector(numVals, true);
			
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
	public void leftMultByRowVector(ColGroupDDC a, MatrixBlock result)
			throws DMLRuntimeException 
	{
		//note: this method is only applicable for numrows < blocksize
		double[] c = result.getDenseBlock();
		final int numCols = getNumCols();
		final int numVals = getNumValues();

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
				for( int i=curRunStartOff; i<curRunStartOff+curRunLen; i++ )
					vsum += a.getData(i, 0);
				curRunEnd = curRunStartOff + curRunLen;
			}
			
			//scale partial results by values and write results
			for( int j = 0; j < numCols; j++ )
				c[ _colIndexes[j] ] += vsum * _values[ valOff+j ];
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
		char[] lbitmap = BitmapEncoder.genRLEBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length+lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length+1);
		rbitmapOffs[rbitmapOffs.length-1] = rbitmaps.length; 
		
		return new ColGroupRLE(_colIndexes, _numRows, loff.length<_numRows,
				rvalues, rbitmaps, rbitmapOffs);
	}

	@Override
	protected final void computeSum(MatrixBlock result, KahanFunction kplus)
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

	@Override
	protected final void computeRowSums(MatrixBlock result, KahanFunction kplus, int rl, int ru)
	{
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
		
		final int numVals = getNumValues();
		double[] c = result.getDenseBlock();
		
		if( ALLOW_CACHE_CONSCIOUS_ROWSUMS 
			&& LOW_LEVEL_OPT && numVals > 1 
			&& _numRows > BitmapEncoder.BITMAP_BLOCK_SZ )
		{
			final int blksz = ColGroupOffset.WRITE_CACHE_BLKSZ/2; 
			
			//step 1: prepare position and value arrays
			
			//current pos / values per RLE list
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			double[] aval = sumAllValues(kplus, kbuff, false);
			
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
						int from = Math.max(bi, start+lstart);
						int to = Math.min(start+lstart+llen,bimax);
						for (int rix=from; rix<to; rix++) {
							kbuff.set(c[2*rix], c[2*rix+1]);
							kplus2.execute2(kbuff, val);
							c[2*rix] = kbuff._sum;
							c[2*rix+1] = kbuff._correction;
						}
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
				double val = sumValues(k, kplus, kbuff);
						
				if (val != 0.0) {
					Pair<Integer,Integer> tmp = skipScanVal(k, rl);
					int bix = tmp.getKey();
					int curRunStartOff = tmp.getValue();
					int curRunEnd = tmp.getValue();
					for ( ; bix<blen && curRunEnd<ru; bix+=2) {
						curRunStartOff = curRunEnd + _data[boff+bix];
						curRunEnd = curRunStartOff + _data[boff+bix+1];
						for (int rix=curRunStartOff; rix<curRunEnd && rix<ru; rix++) {
							kbuff.set(c[2*rix], c[2*rix+1]);
							kplus2.execute2(kbuff, val);
							c[2*rix] = kbuff._sum;
							c[2*rix+1] = kbuff._correction;
						}
					}
				}
			}
		}
	}

	@Override
	protected final void computeColSums(MatrixBlock result, KahanFunction kplus)
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

	@Override
	protected final void computeRowMxx(MatrixBlock result, Builtin builtin, int rl, int ru)
	{
		//NOTE: zeros handled once for all column groups outside
		final int numVals = getNumValues();
		double[] c = result.getDenseBlock();
		
		for (int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			double val = mxxValues(k, builtin);
			
			Pair<Integer,Integer> tmp = skipScanVal(k, rl);
			int bix = tmp.getKey();
			int curRunStartOff = tmp.getValue();
			int curRunEnd = tmp.getValue();
			for(; bix < blen && curRunEnd < ru; bix+=2) {
				curRunStartOff = curRunEnd + _data[boff+bix];
				curRunEnd = curRunStartOff + _data[boff+bix+1];
				for (int rix=curRunStartOff; rix<curRunEnd && rix<ru; rix++)
					c[rix] = builtin.execute2(c[rix], val);
			}
		}
	}
	
	@Override
	public boolean[] computeZeroIndicatorVector()
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
	 * @param numVals number of values
	 * @param rl lower row position
	 * @param astart start positions
	 * @return array of positions for all values
	 */
	private int[] skipScan(int numVals, int rl, int[] astart) {
		int[] apos = allocIVector(numVals, rl==0);
		
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

	private Pair<Integer,Integer> skipScanVal(int k, int rl) {
		int apos = 0; 
		int astart = 0;
		
		if( rl > 0 ) { //rl aligned with blksz	
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
			apos = bix;
			astart = start;
		}
		
		return new Pair<Integer,Integer>(apos, astart);
	}
	
	@Override
	public Iterator<Integer> getIterator(int k) {
		return new RLEValueIterator(k, 0, getNumRows());
	}
	
	@Override
	public Iterator<Integer> getIterator(int k, int rl, int ru) {
		return new RLEValueIterator(k, rl, ru);
	}

	private class RLEValueIterator implements Iterator<Integer>
	{
		private final int _ru;
		private final int _boff;
		private final int _blen;
		private int _bix;
		private int _start;
		private int _rpos;
		
		public RLEValueIterator(int k, int rl, int ru) {
			_ru = ru;
			_boff = _ptr[k];
			_blen = len(k);
			_bix = 0; 
			_start = 0; //init first run
			_rpos = _data[_boff+_bix]; 
			while( _rpos < rl )
				nextRowOffset();
		}

		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}

		@Override
		public Integer next() {
			if( !hasNext() )
				throw new RuntimeException("No more RLE entries.");
			int ret = _rpos;
			nextRowOffset();
			return ret;
		}
		
		private void nextRowOffset() {
			if( !hasNext() )
			  return;
			//get current run information
			int lstart = _data[_boff + _bix]; //start
			int llen = _data[_boff + _bix + 1]; //len
			//advance to next run if necessary
			if( _rpos - _start - lstart + 1 >= llen ) {
				_start += lstart + llen;
				_bix +=2;
				_rpos = (_bix>=_blen) ? _ru : 
					_start + _data[_boff + _bix];
			}
			//increment row index within run
			else {
				_rpos++;
			}
		}
	}
}
