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
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with
 * simple lists of offsets for each set of distinct values.
 * 
 */
public class ColGroupOLE extends ColGroupOffset 
{
	private static final long serialVersionUID = -9157676271360528008L;

	private static final Log LOG = LogFactory.getLog(ColGroupOLE.class.getName());
	
	public ColGroupOLE() {
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
	public ColGroupOLE(int[] colIndices, int numRows, UncompressedBitmap ubm) 
	{
		super(colIndices, numRows, ubm);

		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for( int i=0; i<numVals; i++ ) {
			lbitmaps[i] = BitmapEncoder.genOffsetBitmap(
				ubm.getOffsetsList(i).extractValues(), ubm.getNumOffsets(i));
			totalLen += lbitmaps[i].length;
		}
		
		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);
		

		if( LOW_LEVEL_OPT && CREATE_SKIPLIST
				&& numRows > 2*BitmapEncoder.BITMAP_BLOCK_SZ )
		{
			int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
			_skiplist = new int[numVals];
			int rl = (getNumRows()/2/blksz)*blksz;
			for (int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = 0;
				for( int i=0; i<rl && bix<blen; i+=blksz ) {
					bix += _data[boff+bix] + 1;
				}
				_skiplist[k] = bix;
			}		
		}
		
		//debug output
		double ucSize = MatrixBlock.estimateSizeDenseInMemory(numRows, colIndices.length);
		if( estimateInMemorySize() > ucSize )
			LOG.warn("OLE group larger than UC dense: "+estimateInMemorySize()+" "+ucSize);
	}

	public ColGroupOLE(int[] colIndices, int numRows, boolean zeros, double[] values, char[] bitmaps, int[] bitmapOffs) {
		super(colIndices, numRows, zeros, values);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}
	

	@Override
	public CompressionType getCompType() {
		return CompressionType.OLE_BITMAP;
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) 
	{
		if( LOW_LEVEL_OPT && getNumValues() > 1 )
		{
			final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			
			//cache blocking config and position array
			int[] apos = skipScan(numVals, rl);
					
			//cache conscious append via horizontal scans 
			for( int bi=rl; bi<ru; bi+=blksz ) {
				for (int k = 0, off=0; k < numVals; k++, off+=numCols) {
					int boff = _ptr[k];
					int blen = len(k);					
					int bix = apos[k];
					if( bix >= blen ) 
						continue;
					int len = _data[boff+bix];
					int pos = boff+bix+1;
					for( int i=pos; i<pos+len; i++ )
						for( int j=0, rix = bi+_data[i]; j<numCols; j++ )
							if( _values[off+j]!=0 )
								target.appendValue(rix, _colIndexes[j], _values[off+j]);
					apos[k] += len + 1;
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
			final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final int n = getNumRows();
			
			//cache blocking config and position array
			int[] apos = new int[numVals];					
			int[] cix = new int[numCols];
			
			//prepare target col indexes
			for( int j=0; j<numCols; j++ )
				cix[j] = colixTargets[_colIndexes[j]];
			
			//cache conscious append via horizontal scans 
			for( int bi=0; bi<n; bi+=blksz ) {
				for (int k = 0, off=0; k < numVals; k++, off+=numCols) {
					int boff = _ptr[k];
					int blen = len(k);					
					int bix = apos[k];
					if( bix >= blen ) 
						continue;
					int len = _data[boff+bix];
					int pos = boff+bix+1;
					for( int i=pos; i<pos+len; i++ )
						for( int j=0, rix = bi+_data[i]; j<numCols; j++ )
							if( _values[off+j]!=0 )
								target.appendValue(rix, cix[j], _values[off+j]);
					apos[k] += len + 1;
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
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final int n = getNumRows();
		double[] c = target.getDenseBlock();
		
		//cache blocking config and position array
		int[] apos = allocIVector(numVals, true);
		
		//cache conscious append via horizontal scans 
		int nnz = 0;
		for( int bi=0; bi<n; bi+=blksz ) {
			Arrays.fill(c, bi, Math.min(bi+blksz, n), 0);
			for (int k = 0, off=0; k < numVals; k++, off+=numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if( bix >= blen )
					continue;
				int len = _data[boff+bix];
				int pos = boff+bix+1;
				for( int i=pos; i<pos+len; i++ ) {
					c[bi+_data[i]] = _values[off+colpos];
					nnz++;
				}
				apos[k] += len + 1;
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
			
			//iterate over bitmap blocks and count partial lengths
			int count = 0;
			for (int bix=0; bix < blen; bix+=_data[boff+bix]+1)
				count += _data[boff+bix];
			counts[k] = count;
		}
		
		return counts;
	}
	
	@Override
	public ColGroup scalarOperation(ScalarOperator op)
		throws DMLRuntimeException 
	{
		double val0 = op.executeScalar(0);
		
		//fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if( op.sparseSafe || val0==0 ) {
			return new ColGroupOLE(_colIndexes, _numRows, _zeros, 
					applyScalarOp(op), _data, _ptr);
		}
		
		//slow path: sparse-unsafe operations (potentially create new bitmap)
		//note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if( loff.length==0 ) { //empty offset list: go back to fast path
			return new ColGroupOLE(_colIndexes, _numRows, true,
					applyScalarOp(op), _data, _ptr);
		}
		
		double[] rvalues = applyScalarOp(op, val0, getNumCols());		
		char[] lbitmap = BitmapEncoder.genOffsetBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length+lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length+1);
		rbitmapOffs[rbitmapOffs.length-1] = rbitmaps.length; 
		
		return new ColGroupOLE(_colIndexes, _numRows, loff.length<_numRows,
				rvalues, rbitmaps, rbitmapOffs);
	}

	@Override
	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru)
			throws DMLRuntimeException 
	{
		double[] b = ConverterUtils.getDenseVector(vector);
		double[] c = result.getDenseBlock();
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		
		//prepare reduced rhs w/ relevant values
		double[] sb = new double[numCols];
		for (int j = 0; j < numCols; j++) {
			sb[j] = b[_colIndexes[j]];
		}
		
		if( LOW_LEVEL_OPT && numVals > 1 && _numRows > blksz )
		{
			//since single segment scans already exceed typical L2 cache sizes
			//and because there is some overhead associated with blocking, the
			//best configuration aligns with L3 cache size (x*vcores*64K*8B < L3)
			//x=4 leads to a good yet slightly conservative compromise for single-/
			//multi-threaded and typical number of cores and L3 cache sizes
			final int blksz2 = ColGroupOffset.WRITE_CACHE_BLKSZ;
			
			//step 1: prepare position and value arrays
			int[] apos = skipScan(numVals, rl);
			double[] aval = preaggValues(numVals, sb);
					
			//step 2: cache conscious matrix-vector via horizontal scans 
			for( int bi=rl; bi<ru; bi+=blksz2 ) 
			{
				int bimax = Math.min(bi+blksz2, ru);
				
				//horizontal segment scan, incl pos maintenance
				for (int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					double val = aval[k];
					int bix = apos[k];
					
					for( int ii=bi; ii<bimax && bix<blen; ii+=blksz ) {
						//prepare length, start, and end pos
						int len = _data[boff+bix];
						int pos = boff+bix+1;
						
						//compute partial results
						LinearAlgebraUtils.vectAdd(val, c, _data, pos, ii, len);
						bix += len + 1;
					}

					apos[k] = bix;
				}
			}		
		}
		else
		{	
			//iterate over all values and their bitmaps
			for (int k = 0; k < numVals; k++) 
			{
				//prepare value-to-add for entire value bitmap
				int boff = _ptr[k];
				int blen = len(k);
				double val = sumValues(k, sb);
				
				//iterate over bitmap blocks and add values
				if (val != 0) {
					int bix = 0;
					int off = 0;
					int slen = -1;
					
					//scan to beginning offset if necessary 
					if( rl > 0 ){
						for (; bix<blen & off<rl; bix += slen+1, off += blksz) {
							slen = _data[boff+bix];
						}	
					}
					
					//compute partial results
					for (; bix<blen & off<ru; bix += slen + 1, off += blksz) {
						slen = _data[boff+bix];
						for (int blckIx = 1; blckIx <= slen; blckIx++) {
							c[off + _data[boff+bix + blckIx]] += val;
						}
					}
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
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final int n = getNumRows();
		
		if( LOW_LEVEL_OPT && numVals > 1 && _numRows > blksz )
		{
			//cache blocking config (see matrix-vector mult for explanation)
			final int blksz2 = ColGroupOffset.READ_CACHE_BLKSZ;
			
			//step 1: prepare position and value arrays
			
			//current pos per OLs / output values
			int[] apos = allocIVector(numVals, true);
			double[] cvals = allocDVector(numVals, true);
			
			//step 2: cache conscious matrix-vector via horizontal scans 
			for( int ai=0; ai<n; ai+=blksz2 ) 
			{
				int aimax = Math.min(ai+blksz2, n);
				
				//horizontal segment scan, incl pos maintenance
				for (int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					double vsum = 0;	
					
					for( int ii=ai; ii<aimax && bix<blen; ii+=blksz ) {
						//prepare length, start, and end pos
						int len = _data[boff+bix];
						int pos = boff+bix+1;
						
						//iterate over bitmap blocks and compute partial results (a[i]*1)
						vsum += LinearAlgebraUtils.vectSum(a, _data, ii, pos, len);
						bix += len + 1;
					}

					apos[k] = bix;
					cvals[k] += vsum;
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
				
				//iterate over bitmap blocks and add partial results
				double vsum = 0;
				for (int bix=0, off=0; bix < blen; bix+=_data[boff+bix]+1, off+=blksz)
					vsum += LinearAlgebraUtils.vectSum(a, _data, off, boff+bix+1, _data[boff+bix]);
				
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
		for (int k=0, valOff=0; k<numVals; k++, valOff+=numCols) {
			int boff = _ptr[k];
			
			//iterate over bitmap blocks and add partial results
			double vsum = 0;
			for( int j = boff+1; j < boff+1+_data[boff]; j++ )
				vsum += a.getData(_data[j], 0);
			
			//scale partial results by values and write results
			for( int j = 0; j < numCols; j++ )
				c[ _colIndexes[j] ] += vsum * _values[ valOff+j ];
		}
	}
	
	@Override
	protected final void computeSum(MatrixBlock result, KahanFunction kplus)
	{
		KahanObject kbuff = new KahanObject(result.quickGetValue(0, 0), result.quickGetValue(0, 1));
		
		//iterate over all values and their bitmaps
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		for (int k = 0; k < numVals; k++) 
		{
			int boff = _ptr[k];
			int blen = len(k);
			int valOff = k * numCols;
			
			//iterate over bitmap blocks and count partial lengths
			int count = 0;
			for (int bix=0; bix < blen; bix+=_data[boff+bix]+1)
				count += _data[boff+bix];
			
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
		
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		double[] c = result.getDenseBlock();
		
		if( ALLOW_CACHE_CONSCIOUS_ROWSUMS &&
			LOW_LEVEL_OPT && numVals > 1 && _numRows > blksz )
		{
			final int blksz2 = ColGroupOffset.WRITE_CACHE_BLKSZ/2;
			
			//step 1: prepare position and value arrays
			int[] apos = skipScan(numVals, rl);
			double[] aval = sumAllValues(kplus, kbuff, false);
					
			//step 2: cache conscious row sums via horizontal scans 
			for( int bi=rl; bi<ru; bi+=blksz2 ) 
			{
				int bimax = Math.min(bi+blksz2, ru);
				
				//horizontal segment scan, incl pos maintenance
				for (int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					double val = aval[k];
					int bix = apos[k];
					
					for( int ii=bi; ii<bimax && bix<blen; ii+=blksz ) {
						//prepare length, start, and end pos
						int len = _data[boff+bix];
						int pos = boff+bix+1;
						
						//compute partial results
						for (int i = 0; i < len; i++) {
							int rix = ii + _data[pos + i];
							kbuff.set(c[2*rix], c[2*rix+1]);
							kplus2.execute2(kbuff, val);
							c[2*rix] = kbuff._sum;
							c[2*rix+1] = kbuff._correction;
						}
						bix += len + 1;
					}

					apos[k] = bix;
				}
			}		
		}
		else
		{
			//iterate over all values and their bitmaps
			for (int k = 0; k < numVals; k++) 
			{
				//prepare value-to-add for entire value bitmap
				int boff = _ptr[k];
				int blen = len(k);
				double val = sumValues(k, kplus, kbuff);
				
				//iterate over bitmap blocks and add values
				if (val != 0) {
					int slen;
					int bix = skipScanVal(k, rl);
					for( int off=((rl+1)/blksz)*blksz; bix<blen && off<ru; bix+=slen+1, off+=blksz ) {
						slen = _data[boff+bix];
						for (int i = 1; i <= slen; i++) {
							int rix = off + _data[boff+bix + i];
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
		
		//iterate over all values and their bitmaps
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		for (int k = 0; k < numVals; k++) 
		{
			int boff = _ptr[k];
			int blen = len(k);
			int valOff = k * numCols;
			
			//iterate over bitmap blocks and count partial lengths
			int count = 0;
			for (int bix=0; bix < blen; bix+=_data[boff+bix]+1)
				count += _data[boff+bix];
			
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
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		double[] c = result.getDenseBlock();
		
		//iterate over all values and their bitmaps
		for (int k = 0; k < numVals; k++) 
		{
			//prepare value-to-add for entire value bitmap
			int boff = _ptr[k];
			int blen = len(k);
			double val = mxxValues(k, builtin);
			
			//iterate over bitmap blocks and add values
			int slen;
			int bix = skipScanVal(k, rl);
			for( int off=bix*blksz; bix<blen && off<ru; bix+=slen+1, off+=blksz ) {
				slen = _data[boff+bix];
				for (int i = 1; i <= slen; i++) {
					int rix = off + _data[boff+bix + i];
					c[rix] = builtin.execute2(c[rix], val);
				}
			}
		}
	}
	
	/**
	 * Utility function of sparse-unsafe operations.
	 * 
	 * @return zero indicator vector
	 */
	@Override
	protected boolean[] computeZeroIndicatorVector()
	{
		boolean[] ret = new boolean[_numRows];
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		
		//initialize everything with zero
		Arrays.fill(ret, true);
		
		//iterate over all values and their bitmaps
		for (int k = 0; k < numVals; k++) 
		{
			//prepare value-to-add for entire value bitmap
			int boff = _ptr[k];
			int blen = len(k);
			
			//iterate over bitmap blocks and add values
			int off = 0;
			int slen;
			for( int bix=0; bix < blen; bix+=slen+1, off+=blksz) {
				slen = _data[boff+bix];
				for (int i = 1; i <= slen; i++) {
					ret[off + _data[boff+bix + i]] &= false;
				}
			}
		}
		
		return ret;
	}
	
	@Override
	protected void countNonZerosPerRow(int[] rnnz, int rl, int ru)
	{
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int blksz2 = ColGroupOffset.WRITE_CACHE_BLKSZ;
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		
		//current pos per OLs / output values
		int[] apos = skipScan(numVals, rl);
		
		//cache conscious count via horizontal scans 
		for( int bi=rl; bi<ru; bi+=blksz2 )  {
			int bimax = Math.min(bi+blksz2, ru);
			
			//iterate over all values and their bitmaps
			for (int k = 0; k < numVals; k++)  {
				//prepare value-to-add for entire value bitmap
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				
				//iterate over bitmap blocks and add values
				for( int off=bi, slen=0; bix<blen && off<bimax; bix+=slen+1, off+=blksz ) {
					slen = _data[boff+bix];
					for (int blckIx = 1; blckIx <= slen; blckIx++) {
						rnnz[off + _data[boff+bix + blckIx] - rl] += numCols;
					}
				}
				
				apos[k] = bix;
			}
		}
	}
	
	/////////////////////////////////
	// internal helper functions
	
	/**
	 * Scans to given row_lower position by exploiting any existing 
	 * skip list and scanning segment length fields. Returns array 
	 * of positions for all values.
	 * 
	 * @param numVals number of values
	 * @param rl row lower position
	 * @return array of positions for all values
	 */
	private int[] skipScan(int numVals, int rl) {
		int[] ret = allocIVector(numVals, rl==0);
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		
		if( rl > 0 ) { //rl aligned with blksz		
			int rskip = (getNumRows()/2/blksz)*blksz;
			
			for( int k = 0; k < numVals; k++ ) {
				int boff = _ptr[k];
				int blen = len(k);
				int start = (rl>=rskip)?rskip:0;
				int bix = (rl>=rskip)?_skiplist[k]:0;
				for( int i=start; i<rl && bix<blen; i+=blksz ) {
					bix += _data[boff+bix] + 1;
				}
				ret[k] = bix;
			}
		}
		
		return ret;
	}

	private int skipScanVal(int k, int rl) {
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		
		if( rl > 0 ) { //rl aligned with blksz		
			int rskip = (getNumRows()/2/blksz)*blksz;
			int boff = _ptr[k];
			int blen = len(k);
			int start = (rl>=rskip)?rskip:0;
			int bix = (rl>=rskip)?_skiplist[k]:0;
			for( int i=start; i<rl && bix<blen; i+=blksz ) {
				bix += _data[boff+bix] + 1;
			}
			return bix;
		}
		
		return 0;
	}

	@Override
	public Iterator<Integer> getIterator(int k) {
		return new OLEValueIterator(k, 0, getNumRows());
	}
	
	@Override
	public Iterator<Integer> getIterator(int k, int rl, int ru) {
		return new OLEValueIterator(k, rl, ru);
	}

	private class OLEValueIterator implements Iterator<Integer>
	{
		private final int _ru;
		private final int _boff;
		private final int _blen;
		private int _bix;
		private int _start;
		private int _slen;
		private int _spos;
		private int _rpos;
		
		public OLEValueIterator(int k, int rl, int ru) {
			_ru = ru;
			_boff = _ptr[k];
			_blen = len(k);
			
			//initialize position via segment-aligned skip-scan
			int lrl = rl - rl%BitmapEncoder.BITMAP_BLOCK_SZ;
			_bix = skipScanVal(k, lrl);
			_start = lrl; 
			
			//move position to actual rl boundary
			if( _bix < _blen ) {
				_slen = _data[_boff + _bix];
				_spos = 0;
				_rpos = _data[_boff + _bix + 1];
				while( _rpos < rl )
					nextRowOffset();
			}
			else {
				_rpos = _ru;
			}
		}

		@Override
		public boolean hasNext() {
			return (_rpos < _ru);
		}

		@Override
		public Integer next() {
			if( !hasNext() )
				throw new RuntimeException("No more OLE entries.");
			int ret = _rpos;
			nextRowOffset();
			return ret;
		}
		
		private void nextRowOffset() {
			if( _spos+1 < _slen ) {
				_spos++;
				_rpos = _start + _data[_boff + _bix + _spos + 1];
			}
			else {
				_start += BitmapEncoder.BITMAP_BLOCK_SZ;
				_bix += _slen+1;
				if( _bix < _blen ) {
					_slen = _data[_boff + _bix];
					_spos = 0;
					_rpos = _start + _data[_boff + _bix + 1];
				}
				else {
					_rpos = _ru;
				}
			}
		}
	}
}
