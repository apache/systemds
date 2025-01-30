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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.lib.CLALibRexpand;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.RevIndex;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.SortUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

/**
 * MB:
 * Library for selected matrix reorg operations including special cases
 * and all combinations of dense and sparse representations.
 * 
 * Current list of supported operations:
 *  - reshape, 
 *  - r' (transpose), 
 *  - rdiag (diagV2M/diagM2V), 
 *  - rsort (sorting data/indexes)
 *  - rmempty (remove empty)
 *  - rexpand (outer/table-seq expansion)
 */
public class LibMatrixReorg {

	protected static final Log LOG = LogFactory.getLog(LibMatrixReorg.class.getName());

	//minimum number of elements for multi-threaded execution
	public static long PAR_NUMCELL_THRESHOLD = 1024*1024; //1M
	
	// SORTING threshold
	public static final int PAR_NUMCELL_THRESHOLD_SORT = 1024;
	
	//allow shallow dense/sparse copy for unchanged data (which is 
	//safe due to copy-on-write and safe update-in-place handling)
	public static final boolean SHALLOW_COPY_REORG = true;
	
	//use csr instead of mcsr sparse block for rexpand columns / diag v2m
	public static final boolean SPARSE_OUTPUTS_IN_CSR = true;

	//use legacy in-place transpose dense instead of brenner in-place transpose dense
	public static final boolean TRANSPOSE_IN_PLACE_DENSE_LEGACY = true;
	
	private enum ReorgType {
		TRANSPOSE,
		REV,
		ROLL,
		DIAG,
		RESHAPE,
		SORT,
		INVALID,
	}
	
	private LibMatrixReorg() {
		//prevent instantiation via private constructor
	}
	
	/////////////////////////
	// public interface    //
	/////////////////////////

	public static boolean isSupportedReorgOperator( ReorgOperator op ) {
		return (getReorgType(op) != ReorgType.INVALID);
	}

	public static MatrixBlock reorg( MatrixBlock in, MatrixBlock out, ReorgOperator op ) {
		ReorgType type = getReorgType(op);

		switch( type ) {
			case TRANSPOSE:
				if( op.getNumThreads() > 1 )
					return transpose(in, out, op.getNumThreads());
				else
					return transpose(in, out);
			case REV:
				return rev(in, out);
			case ROLL:
				RollIndex rix = (RollIndex) op.fn;
				return roll(in, out, rix.getShift());
			case DIAG:
				return diag(in, out);
			case SORT:
				SortIndex ix = (SortIndex) op.fn;
				if (op.getNumThreads() > 1)
					return sort(in, out, ix.getCols(), ix.getDecreasing(), ix.getIndexReturn(), op.getNumThreads());
				else
					return sort(in, out, ix.getCols(), ix.getDecreasing(), ix.getIndexReturn());
			default:
				throw new DMLRuntimeException("Unsupported reorg operator: "+op.fn);
		}
	}

	public static MatrixBlock reorgInPlace(MatrixBlock in, ReorgOperator op){
		ReorgType type = getReorgType(op);
		switch (type){
			case TRANSPOSE:
				return transposeInPlace(in, op.getNumThreads());
			case REV:
			case ROLL:
			case SORT:
				throw new DMLRuntimeException("Not implemented inplace: " + op.fn.getClass().getSimpleName());
			default:
				throw new DMLRuntimeException("Unsupported inplace reorg operator: " + op.fn.getClass().getSimpleName());
		}
	}

	public static MatrixBlock transpose(MatrixBlock in) {
		final int clen = in.getNumColumns();
		final int rlen = in.getNumRows();
		final long nnz = in.getNonZeros();
		final boolean sparseOut = MatrixBlock.evalSparseFormatInMemory(clen, rlen, nnz, true);
		return transpose(in, new MatrixBlock(clen, rlen, sparseOut));
	}

	public static MatrixBlock transpose( MatrixBlock in, MatrixBlock out ) {
		if(in instanceof CompressedMatrixBlock)
			throw new DMLCompressionException("Invalid call to transposed with a compressed matrix block");
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return out;
	
		//set basic meta data
		out.nonZeros = in.nonZeros;
		
		//shallow dense vector transpose (w/o result allocation)
		//since the physical representation of dense vectors is always the same,
		//we don't need to create a copy, given our copy on write semantics.
		//however, note that with update in-place this would be an invalid optimization
		if( SHALLOW_COPY_REORG && !in.sparse && !out.sparse && (in.rlen==1 || in.clen==1)  ) {
			out.denseBlock = DenseBlockFactory.createDenseBlock(in.getDenseBlockValues(), in.clen, in.rlen);
			return out;
		}
		
		// Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		if( out.sparse )
			out.allocateSparseRowsBlock(false);
		else
			out.allocateDenseBlock(false);
	
		//execute transpose operation
		boolean ultraSparse = (in.sparse && out.sparse && in.nonZeros < Math.max(in.rlen, in.clen));
		if( !in.sparse && !out.sparse )
			transposeDenseToDense(in, out, 0, in.rlen, 0, in.clen);
		else if( ultraSparse )
			transposeUltraSparse(in, out);
		else if( in.sparse && out.sparse )
			transposeSparseToSparse(in, out, 0, in.rlen, 0, in.clen, 
				countNnzPerColumn(in, 4096));
		else if( in.sparse )
			transposeSparseToDense(in, out, 0, in.rlen, 0, in.clen);
		else
			transposeDenseToSparse(in, out);
		
		// System.out.println("r' ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}

	public static MatrixBlock transpose(MatrixBlock in, int k) {
		return transpose(in, k, false);
	}

	public static MatrixBlock transpose(MatrixBlock in, int k, boolean allowCSR) {
		final int clen = in.getNumColumns();
		final int rlen = in.getNumRows();
		final long nnz = in.getNonZeros();
		final boolean sparseOut = MatrixBlock.evalSparseFormatInMemory(clen, rlen, nnz, allowCSR);
		return transpose(in, new MatrixBlock(clen, rlen, sparseOut), k, allowCSR);
	}


	public static MatrixBlock transpose( MatrixBlock in, MatrixBlock out, int k ) {
		return transpose(in, out, k, false);
	}
	
	public static MatrixBlock transpose(MatrixBlock in, MatrixBlock out, int k, boolean allowCSR) {
		// redirect small or special cases to sequential execution
		if(in.isEmptyBlock(false) //
			|| ((long) in.rlen * (long) in.clen < PAR_NUMCELL_THRESHOLD) //
			|| k <= 1 //
			|| (SHALLOW_COPY_REORG && !in.sparse && !out.sparse && (in.rlen == 1 || in.clen == 1)) //
			|| (in.sparse && !out.sparse && in.rlen == 1) //
			|| (!in.sparse && out.sparse && in.rlen == 1) //
			|| (in.sparse && out.sparse && in.isUltraSparse(false)))
		{
			return transpose(in, out);
		}
		// set meta data and allocate output arrays (if required)
		out.nonZeros = in.nonZeros;

		if(!in.sparse && out.sparse){
			// special case dense to sparse is different than others because appending to sparse rows.
			transposeDenseToSparse(in, out, k);
			return out;
		}

		// Timing time = new Timing(true);

		// CSR is only allowed in the transposed output if the number of non zeros is counted in the columns
		// and the temporary count arrays are not larger than the entire input
		allowCSR = allowCSR && (in.clen <= 4096 || out.nonZeros < 10000000) 
				&& (k*4*in.clen < in.getInMemorySize());
		
		int[] cnt = null;
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			if(out.sparse && allowCSR) {
				final int size = (int) out.nonZeros;
				final Future<int[]> f = countNNZColumns(in, k, pool);
				out.sparseBlock = new SparseBlockCSR(in.getNumColumns(), size, size);
				final int[] outPtr = ((SparseBlockCSR) out.sparseBlock).rowPointers();
				
				// pre-processing (compute nnz per column once for sparse)
				// filter matrices with many columns since the CountNnzTask would return
				// null if the number of columns is larger than threshold
				cnt = f.get();
				for(int i = 0; i < cnt.length; i++) {
					// set out pointers to correct start of rows.
					outPtr[i + 1] = outPtr[i] + cnt[i];
					// set the cnt value to the new pointer to start of row in CSR
					cnt[i] = outPtr[i];
				}
			}
			else if(out.sparse)
				out.allocateSparseRowsBlock(false);
			else
				out.allocateDenseBlock(false);

			// compute actual transpose and check for errors
			ArrayList<TransposeTask> tasks = new ArrayList<>();
			boolean allowReturnBlock = out.sparse && in.sparse 
				&& in.rlen >= in.clen && cnt == null && !in.isUltraSparse(false);
			boolean row = (in.sparse || in.rlen >= in.clen)
				&& (!out.sparse || allowReturnBlock) && !in.isUltraSparse(false);
			int len = row ? in.rlen : in.clen;
			int blklen = (int) (Math.ceil((double) len / k));
			blklen += (!out.sparse && (blklen % 8) != 0) ? 8 - blklen % 8 : 0;
			blklen = (in.sparse) ? Math.max(blklen, 32) : blklen;

			for(int i = 0; i < k & i * blklen < len; i++)
				tasks.add(new TransposeTask(in, out, row, i * blklen, Math.min((i + 1) * blklen, len), cnt, allowReturnBlock));
			
			if(allowReturnBlock) {
				List<MatrixBlock> blocks = new ArrayList<>();
				for(Future<MatrixBlock> task : pool.invokeAll(tasks)) {
					MatrixBlock m = task.get();
					if(allowReturnBlock && m != null)
						blocks.add(m);
				}

				if(allowReturnBlock)
					combine(blocks, out, row, k);
			}
			else {
				for(Future<MatrixBlock> task : pool.invokeAll(tasks)) {
					task.get();
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}

		// System.out.println("r' k="+k+" ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}

	private static void combine(List<MatrixBlock> blocks, MatrixBlock out, boolean row, int k){
		MatrixBlock.append(blocks, out, row, k);
	}

	public static Future<int[]> countNNZColumns(MatrixBlock in, int k, ExecutorService pool)
		throws InterruptedException, ExecutionException {
		final List<Future<int[]>> rtasks = countNNZColumnsFuture(in, k, pool);
		return pool.submit(() -> {
			int[] cnt = null;
			for(Future<int[]> rtask : rtasks)
				cnt = mergeNnzCounts(cnt, rtask.get());
			return cnt;
		});
	}

	public static List<Future<int[]>> countNNZColumnsFuture(MatrixBlock in, int k, ExecutorService pool) throws InterruptedException {
		ArrayList<CountNnzTask> tasks = new ArrayList<>();
		int blklen = (int) (Math.ceil((double) in.rlen / k));
		for(int i = 0; i < k & i * blklen < in.rlen; i++)
			tasks.add(new CountNnzTask(in, i * blklen, Math.min((i + 1) * blklen, in.rlen)));
		return pool.invokeAll(tasks);
	}

	public static MatrixBlock transposeInPlace(MatrixBlock in, int k){
		// Timing time = new Timing(true);
		MatrixBlock out = null;
		if(in.isEmpty()) 
			out = new MatrixBlock(in.getNumColumns(), in.getNumRows(), true);
		else if(in.isInSparseFormat()) {
			// If input is sparse use default implementation and allocate a new matrix.
			out = transpose(in, new MatrixBlock(in.getNumColumns(), in.getNumRows(), true), k, true);
		}
		else {
			transposeInPlaceDense(in, k);
			out = in;
		}
		// System.out.println("r' in place k="+k+" ("+in.rlen+", "+in.clen+") in "+time.stop()+" ms.");
		return out;
	}

	public static MatrixBlock rev( MatrixBlock in, MatrixBlock out ) {
		//Timing time = new Timing(true);
	
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return out;
		
		//special case: row vector
		if( in.rlen == 1 ) {
			out.copy(in);
			return out;
		}
		
		if( in.sparse )
			reverseSparse( in, out );
		else
			reverseDense( in, out );
		
		//System.out.println("rev ("+in.rlen+", "+in.clen+", "+in.sparse+") in "+time.stop()+" ms.");

		return out;
	}

	public static void rev( IndexedMatrixValue in, long rlen, int blen, ArrayList<IndexedMatrixValue> out ) {
		//input block reverse 
		MatrixIndexes inix = in.getIndexes();
		MatrixBlock inblk = (MatrixBlock) in.getValue(); 
		MatrixBlock tmpblk = rev(inblk, new MatrixBlock(inblk.getNumRows(), inblk.getNumColumns(), inblk.isInSparseFormat()));
		
		//split and expand block if necessary (at most 2 blocks)
		if( rlen % blen == 0 ) //special case: aligned blocks 
		{
			int nrblks = (int)Math.ceil((double)rlen/blen);
			out.add(new IndexedMatrixValue(
					new MatrixIndexes(nrblks-inix.getRowIndex()+1, inix.getColumnIndex()), tmpblk));
		}
		else //general case: unaligned blocks
		{
			//compute target positions and sizes
			long pos1 = rlen - UtilFunctions.computeCellIndex(inix.getRowIndex(), blen, tmpblk.getNumRows()-1) + 1;
			long pos2 = pos1 + tmpblk.getNumRows() - 1;
			int ipos1 = UtilFunctions.computeCellInBlock(pos1, blen);
			int iposCut = tmpblk.getNumRows() - ipos1 - 1;
			int blkix1 = (int)UtilFunctions.computeBlockIndex(pos1, blen);
			int blkix2 = (int)UtilFunctions.computeBlockIndex(pos2, blen);
			int blklen1 = UtilFunctions.computeBlockSize(rlen, blkix1, blen);
			int blklen2 = UtilFunctions.computeBlockSize(rlen, blkix2, blen);
			
			//slice first block
			MatrixIndexes outix1 = new MatrixIndexes(blkix1, inix.getColumnIndex());
			MatrixBlock outblk1 = new MatrixBlock(blklen1, inblk.getNumColumns(), inblk.isInSparseFormat());
			MatrixBlock tmp1 = tmpblk.slice(0, iposCut);
			outblk1.leftIndexingOperations(tmp1, ipos1, ipos1+tmp1.getNumRows()-1,
				0, tmpblk.getNumColumns()-1, outblk1, UpdateType.INPLACE_PINNED);
			out.add(new IndexedMatrixValue(outix1, outblk1));
			
			//slice second block (if necessary)
			if( blkix1 != blkix2 ) {
				MatrixIndexes outix2 = new MatrixIndexes(blkix2, inix.getColumnIndex());
				MatrixBlock outblk2 = new MatrixBlock(blklen2, inblk.getNumColumns(), inblk.isInSparseFormat());
				MatrixBlock tmp2 = tmpblk.slice(iposCut+1, tmpblk.getNumRows()-1);
				outblk2.leftIndexingOperations(tmp2, 0, tmp2.getNumRows()-1, 0, tmpblk.getNumColumns()-1, outblk2, UpdateType.INPLACE_PINNED);
				out.add(new IndexedMatrixValue(outix2, outblk2));
			}
		}
	}

	public static MatrixBlock roll(MatrixBlock in, MatrixBlock out, int shift) {
		//sparse-safe operation
		if (in.isEmptyBlock(false))
			return out;

		//special case: row vector
		if (in.rlen == 1) {
			out.copy(in);
			return out;
		}

		if (in.sparse)
			rollSparse(in, out, shift);
		else
			rollDense(in, out, shift);

		return out;
	}

	public static void roll(IndexedMatrixValue in, long rlen, int blen, int shift, ArrayList<IndexedMatrixValue> out) {
		MatrixIndexes inMtxIdx = in.getIndexes();
		MatrixBlock inMtxBlk = (MatrixBlock) in.getValue();
		shift %= ((rlen != 0) ? (int) rlen : 1); // Handle row length boundaries for shift

		long inRowIdx = UtilFunctions.computeCellIndex(inMtxIdx.getRowIndex(), blen, 0) - 1;

		int totalCopyLen = 0;
		while (totalCopyLen < inMtxBlk.getNumRows()) {
			// Calculate row and block index for the current part
			long outRowIdx = (inRowIdx + shift) % rlen;
			long outBlkIdx = UtilFunctions.computeBlockIndex(outRowIdx + 1, blen);
			int outBlkLen = UtilFunctions.computeBlockSize(rlen, outBlkIdx, blen);
			int outRowIdxInBlk = (int) (outRowIdx % blen);

			// Calculate copy length
			int copyLen = Math.min((int) (outBlkLen - outRowIdxInBlk), inMtxBlk.getNumRows() - totalCopyLen);

			// Create the output block and copy data
			MatrixIndexes outMtxIdx = new MatrixIndexes(outBlkIdx, inMtxIdx.getColumnIndex());
			MatrixBlock outMtxBlk = new MatrixBlock(outBlkLen, inMtxBlk.getNumColumns(), inMtxBlk.isInSparseFormat());
			copyMtx(inMtxBlk, outMtxBlk, totalCopyLen, outRowIdxInBlk, copyLen, false, false);
			out.add(new IndexedMatrixValue(outMtxIdx, outMtxBlk));

			// Update counters for next iteration
			totalCopyLen += copyLen;
			inRowIdx += totalCopyLen;
		}
	}

	public static MatrixBlock diag( MatrixBlock in, MatrixBlock out ) {
		//Timing time = new Timing(true);
		
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return out;
		
		int rlen = in.rlen;
		int clen = in.clen;
		
		if( clen == 1 ) //diagV2M
			diagV2M( in, out );
		else if ( rlen == clen ) //diagM2V
			diagM2V( in, out );
		else
			throw new DMLRuntimeException("Reorg diagM2V requires squared block input. ("+rlen+", "+clen+")");
		
		//System.out.println("rdiag ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}

	public static MatrixBlock sort(MatrixBlock in, MatrixBlock out, int[] by, boolean desc, boolean ixret) {
		return sort(in,out,by,desc,ixret,1);
	}

	/**
	 * @param in Input matrix to sort
	 * @param out Output matrix where the sorted input is inserted to
	 * @param by The Ordering parameter
	 * @param desc A boolean, specifying if it should be descending order.
	 * @param ixret A boolean, specifying if the return should be the sorted indexes.
	 * @param k Number of parallel threads
	 * @return The sorted out matrix.
	 */
	public static MatrixBlock sort(MatrixBlock in, MatrixBlock out, int[] by, boolean desc, boolean ixret, int k) {
		//Timing time = new Timing(true);
		
		//meta data gathering and preparation
		boolean sparse = in.isInSparseFormat();
		int rlen = in.rlen;
		int clen = in.clen;
		out.sparse = (in.sparse && !ixret);
		out.nonZeros = ixret ? rlen : in.nonZeros;

		//step 1: error handling
		if( !isValidSortByList(by, clen) )
			throw new DMLRuntimeException("Sort configuration issue: invalid orderby columns: "
				+ Arrays.toString(by)+" ("+rlen+"x"+clen+" input).");
		
		//step 2: empty block / special case handling
		if( !ixret ) //SORT DATA
		{
			if( in.isEmptyBlock(false) ) //EMPTY INPUT BLOCK
				return out;
			
			if( !sparse && clen == 1 ) { //DENSE COLUMN VECTOR
				//in-place quicksort, unstable (no indexes needed)
				out.copy( in ); //dense (always single block)
				if (k > 1)
					Arrays.parallelSort(out.getDenseBlockValues());
				else 
					Arrays.sort(out.getDenseBlockValues());
				if( desc )
					sortReverseDense(out);
				return out;
			}
		}
		else //SORT INDEX
		{
			if( in.isEmptyBlock(false) ) { //EMPTY INPUT BLOCK
				out.allocateDenseBlock(false); //single block
				double[] c = out.getDenseBlockValues();
				// create a list containing the indexes of each element in in.
				for( int i=0; i<rlen; i++ )
					c[i] = i+1; //seq(1,n)
				return out;
			}
		}

		//step 3: index vector sorting
		//create index vector and extract values
		//TODO perf: reconsider partition sort to avoid unnecessary barriers
		int[] vix = new int[rlen];
		double[] values = new double[rlen];
		for( int i=0; i<rlen; i++ ) {
			vix[i] = i;
			values[i] = in.get(i, by[0]-1);
		}

		// step 4: split the data into number of blocks of PAR_NUMCELL_THRESHOLD_SORT (1024) elements.
		if (k == 1 || rlen < PAR_NUMCELL_THRESHOLD_SORT){ // There is no parallel
			//sort index vector on extracted data (unstable)
			SortUtils.sortByValue(0, rlen, values, vix);
		} else {
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<SortTask> tasks = new ArrayList<>();
				
				// sort smaller blocks.
				int blklen = (int)(Math.ceil((double)rlen/k));
				for( int i=0; i*blklen<rlen; i++ ){
					int start = i*blklen;
					int stop = Math.min(rlen , i*blklen + blklen);
					tasks.add(new SortTask(start, stop, vix, values));
				}
				CommonThreadPool.invokeAndShutdown(pool, tasks);
				mergeSortedBlocks(blklen, vix, values, k);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}


		//sort by secondary columns if required (in-place)
		if( by.length > 1 )
			sortBySecondary(0, rlen, values, vix, in, by, 1);

		//flip order if descending requested (note that this needs to happen
		//before we ensure stable outputs, hence we also flip values)
		if(desc) {
			sortReverseDense(vix);
			sortReverseDense(values);
		}

		//final pass to ensure stable output
		sortIndexesStable(0, rlen, values, vix, in, by, 1);

		//step 4: create output matrix (guaranteed non-empty, see step 2)
		if( !ixret ) {
			out.allocateBlock();
			//copy input data in sorted order into result
			if(k > 1){

				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<CopyTask> tasks = new ArrayList<>();
				ArrayList<Integer> blklen = UtilFunctions
					.getBalancedBlockSizesDefault(rlen, k, false);
				for( int i=0, lb=0; i<blklen.size(); lb+=blklen.get(i), i++ )
					tasks.add( new CopyTask(in, out, vix, lb, lb+blklen.get(i)));
				CommonThreadPool.invokeAndShutdown(pool, tasks);
			}
			else{
				ArrayList<Integer> blklen = UtilFunctions
				.getBalancedBlockSizesDefault(rlen, k, false);
				for( int i=0, lb=0; i<blklen.size(); lb+=blklen.get(i), i++ )
					new CopyTask(in, out, vix, lb, lb+blklen.get(i)).call();
			}
		}
		else {
			//copy sorted index vector into result
			out.allocateDenseBlock(false);
			DenseBlock c = out.getDenseBlock();
			for( int i=0; i<rlen; i++ )
				c.set(i, 0, vix[i]+1);
		}
		
		return out;
	}
	
	/**
	 * CP reshape operation (single input, single output matrix)
	 * 
	 * NOTE: In contrast to R, the rowwise parameter specifies both the read and write order, with row-wise being the
	 * default, while R uses always a column-wise read, rowwise specifying the write order and column-wise being the
	 * default.
	 * 
	 * @param in      input matrix
	 * @param rows    number of rows
	 * @param cols    number of columns
	 * @param rowwise if true, reshape by row
	 * @return output matrix
	 */
	public static MatrixBlock reshape(MatrixBlock in, int rows, int cols, boolean rowwise) {
		return reshape(in, null, rows,cols, rowwise, 1);
	}

	/**
	 * CP reshape operation (single input, single output matrix)
	 * 
	 * NOTE: In contrast to R, the rowwise parameter specifies both the read and write order, with row-wise being the
	 * default, while R uses always a column-wise read, rowwise specifying the write order and column-wise being the
	 * default.
	 * 
	 * @param in      input matrix
	 * @param out     output matrix
	 * @param rows    number of rows
	 * @param cols    number of columns
	 * @param rowwise if true, reshape by row
	 * @return output matrix
	 */
	public static MatrixBlock reshape(MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise) {
		return reshape(in, out, rows,cols, rowwise, 1);
	}

	/**
	 * CP reshape operation (single input, single output matrix)
	 * 
	 * NOTE: In contrast to R, the rowwise parameter specifies both the read and write order, with row-wise being the
	 * default, while R uses always a column-wise read, rowwise specifying the write order and column-wise being the
	 * default.
	 * 
	 * @param in      input matrix
	 * @param out     output matrix
	 * @param rows    number of rows
	 * @param cols    number of columns
	 * @param rowwise if true, reshape by row
	 * @param k       The parallelization degree
	 * @return output matrix
	 */
	public static MatrixBlock reshape(MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise, int k) {
		try{
			final int rlen = in.rlen;
			final int clen = in.clen;
			
			if(out == null)
				out = new MatrixBlock();
	
			//check for same dimensions
			if(rlen == rows && clen == cols) {
				// copy incl dims, nnz
				if(SHALLOW_COPY_REORG)
					out.copyShallow(in);
				else
					out.copy(in);
				return out;
			}
	
			//check validity
			if(((long) rlen) * clen != ((long) rows) * cols)
				throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells (" + rlen + ":"
					+ clen + ", " + rows + ":" + cols + ").");
		
			//determine output representation
			out.sparse = MatrixBlock.evalSparseFormatInMemory(rows, cols, in.nonZeros);
			
			//set output dimensions
			out.rlen = rows;
			out.clen = cols;
			out.nonZeros = in.nonZeros;
			
			//core reshape (sparse or dense)
			if(!in.sparse && !out.sparse)
				reshapeDense(in, out, rows, cols, rowwise);
			else if(in.sparse && out.sparse)
				reshapeSparse(in, out, rows, cols, rowwise, k);
			else if(in.sparse)
				reshapeSparseToDense(in, out, rows, cols, rowwise);
			else
				reshapeDenseToSparse(in, out, rows, cols, rowwise);
			
			return out;
		}
		catch(Exception e) {
			throw new RuntimeException("Failed to reshape Matrix", e);
		}
	}


	/**
	 * MR/SPARK reshape interface - for reshape we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 * 
	 * @param in indexed matrix value
	 * @param mcIn input matrix characteristics
	 * @param mcOut output matrix characteristics
	 * @param rowwise if true, reshape by row
	 * @param outputEmptyBlocks output blocks with nnz=0
	 * @return list of indexed matrix values
	 */
	public static List<IndexedMatrixValue> reshape(IndexedMatrixValue in, DataCharacteristics mcIn,
	                                               DataCharacteristics mcOut, boolean rowwise, boolean outputEmptyBlocks ) {
		//prepare inputs
		MatrixIndexes ixIn = in.getIndexes();
		MatrixBlock mbIn = (MatrixBlock) in.getValue();
		
		//prepare result blocks (no reuse in order to guarantee mem constraints)
		Collection<MatrixIndexes> rix = computeAllResultBlockIndexes(ixIn, mcIn, mcOut, mbIn, rowwise, outputEmptyBlocks);
		Map<MatrixIndexes, MatrixBlock> rblk = createAllResultBlocks(rix, mbIn.nonZeros, mcOut);
		
		//basic algorithm
		long row_offset = (ixIn.getRowIndex()-1)*mcIn.getBlocksize();
		long col_offset = (ixIn.getColumnIndex()-1)*mcIn.getBlocksize();
		if( mbIn.sparse )
			reshapeSparse(mbIn, row_offset, col_offset, rblk, mcIn, mcOut, rowwise);
		else //dense
			reshapeDense(mbIn, row_offset, col_offset, rblk, mcIn, mcOut, rowwise);
		
		//prepare output (sparsity switch, wrapper)
		return rblk.entrySet().stream()
			.filter( e -> outputEmptyBlocks || !e.getValue().isEmptyBlock(false))
			.map(e -> {e.getValue().examSparsity(); return new IndexedMatrixValue(e.getKey(),e.getValue());})
			.collect(Collectors.toList());
	}

	/**
	 * CP rmempty operation (single input, single output matrix) 
	 * 
	 * @param in input matrix
	 * @param ret output matrix
	 * @param rows ?
	 * @param emptyReturn return row/column of zeros for empty input
	 * @param select ?
	 * @return matrix block
	 */
	public static MatrixBlock rmempty(MatrixBlock in, MatrixBlock ret, boolean rows, boolean emptyReturn, MatrixBlock select) {
		//check for empty inputs 
		//(the semantics of removeEmpty are that for an empty m-by-n matrix, the output 
		//is an empty 1-by-n or m-by-1 matrix because we don't allow matrices with dims 0)
		if( in.isEmptyBlock(false) && select == null  ) {
			int n = emptyReturn ? 1 : 0;
			if( rows )
				ret.reset(n, in.clen, in.sparse);
			else //cols
				ret.reset(in.rlen, n, in.sparse);
			return ret;
		}
		
		// short-circuit for select-all (shallow-copy input)
		if( select != null && (select.nonZeros == (rows?in.rlen:in.clen)) ) {
			return in;
		}
		
		// core removeEmpty
		if( rows )
			return removeEmptyRows(in, ret, select, emptyReturn);
		else //cols
			return removeEmptyColumns(in, ret, select, emptyReturn);
	}

	/**
	 * MR rmempty interface - for rmempty we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 * 
	 * @param data ?
	 * @param offset ?
	 * @param rmRows ?
	 * @param len ?
	 * @param blen block length
	 * @param outList list of indexed matrix values
	 */
	public static void rmempty(IndexedMatrixValue data, IndexedMatrixValue offset, boolean rmRows, long len, long blen, ArrayList<IndexedMatrixValue> outList) {
		//sanity check inputs
		if( !(data.getValue() instanceof MatrixBlock && offset.getValue() instanceof MatrixBlock) )
			throw new DMLRuntimeException("Unsupported input data: expected "+MatrixBlock.class.getName()+" but got "+data.getValue().getClass().getName()+" and "+offset.getValue().getClass().getName());
		if(     rmRows && data.getValue().getNumRows()!=offset.getValue().getNumRows() 
			|| !rmRows && data.getValue().getNumColumns()!=offset.getValue().getNumColumns()  ){
			throw new DMLRuntimeException("Dimension mismatch between input data and offsets: ["
					+data.getValue().getNumRows()+"x"+data.getValue().getNumColumns()+" vs "+offset.getValue().getNumRows()+"x"+offset.getValue().getNumColumns());
		}
		
		//compute outputs (at most two output blocks)
		HashMap<MatrixIndexes,IndexedMatrixValue> out = new HashMap<>();
		MatrixBlock linData = (MatrixBlock) data.getValue();
		MatrixBlock linOffset = (MatrixBlock) offset.getValue();
		MatrixIndexes tmpIx = new MatrixIndexes(-1,-1);
		if( rmRows ) //margin = "rows"
		{
			long rlen = len; //max dimensionality
			long clen = linData.getNumColumns();
			
			for( int i=0; i<linOffset.getNumRows(); i++ ) {
				long rix = (long)linOffset.get(i, 0);
				if( rix <= 0 || rix > rlen ) //skip empty row / cut-off rows
					continue;
				
				//get single row from source block
				MatrixBlock src = linData.slice(i, i, 0, (int)(clen-1), new MatrixBlock());
				long brix = (rix-1)/blen+1;
				long lbrix = (rix-1)%blen;
				tmpIx.setIndexes(brix, data.getIndexes().getColumnIndex());
				 //create target block if necessary
				if( !out.containsKey(tmpIx) ) {
					IndexedMatrixValue tmpIMV = new IndexedMatrixValue(new MatrixIndexes(),new MatrixBlock());
					tmpIMV.getIndexes().setIndexes(tmpIx);
					((MatrixBlock)tmpIMV.getValue()).reset((int)Math.min(blen, rlen-((brix-1)*blen)), (int)clen);
					out.put(tmpIMV.getIndexes(), tmpIMV);
				}
				//put single row into target block
				((MatrixBlock)out.get(tmpIx).getValue()).copy(
					(int)lbrix, (int)lbrix, 0, (int)clen-1, src, false);
			}
		}
		else //margin = "cols"
		{
			long rlen = linData.getNumRows();
			long clen = len;
			
			for( int i=0; i<linOffset.getNumColumns(); i++ ) {
				long cix = (long)linOffset.get(0, i);
				if( cix <= 0 || cix > clen ) //skip empty col / cut-off cols
					continue;
				
				//get single row from source block
				MatrixBlock src = linData.slice(0, (int)(rlen-1), i, i, new MatrixBlock());
				long bcix = (cix-1)/blen+1;
				long lbcix = (cix-1)%blen;
				tmpIx.setIndexes(data.getIndexes().getRowIndex(), bcix);
				 //create target block if necessary
				if( !out.containsKey(tmpIx) ) {
					IndexedMatrixValue tmpIMV = new IndexedMatrixValue(new MatrixIndexes(),new MatrixBlock());
					tmpIMV.getIndexes().setIndexes(tmpIx);
					((MatrixBlock)tmpIMV.getValue()).reset((int)rlen,(int)Math.min(blen, clen-((bcix-1)*blen)));
					out.put(tmpIMV.getIndexes(), tmpIMV);
				}
				//put single row into target block
				((MatrixBlock)out.get(tmpIx).getValue()).copy(
					0, (int)rlen-1, (int)lbcix, (int)lbcix, src, false);
			}
		}
		
		//prepare and return outputs (already in cached values)
		for( IndexedMatrixValue imv : out.values() ){
			((MatrixBlock)imv.getValue()).recomputeNonZeros();
			outList.add(imv);
		}
	}

	/**
	 * CP rexpand operation (single input, single output), the classic example of this operation is one hot encoding of a
	 * column to multiple columns.
	 * 
	 * @param in     Input matrix
	 * @param ret    Output matrix
	 * @param max    Number of rows/cols of the output
	 * @param rows   If the expansion is in rows direction
	 * @param cast   If the values contained should be cast to double (rounded up and down)
	 * @param ignore Ignore if the input contain values below zero that technically is incorrect input.
	 * @param k      Degree of parallelism
	 * @return Output matrix rexpanded
	 */
	public static MatrixBlock rexpand(MatrixBlock in, MatrixBlock ret, double max, boolean rows, boolean cast, boolean ignore, int k) {
		return rexpand(in, ret, UtilFunctions.toInt(max), rows, cast, ignore, k);
	}

	/**
	 * CP rexpand operation (single input, single output), the classic example of this operation is one hot encoding of a
	 * column to multiple columns.
	 * 
	 * @param in     Input matrix
	 * @param ret    Output matrix
	 * @param max    Number of rows/cols of the output
	 * @param rows   If the expansion is in rows direction
	 * @param cast   If the values contained should be cast to double (rounded up and down)
	 * @param ignore Ignore if the input contain values below zero that technically is incorrect input.
	 * @param k      Degree of parallelism
	 * @return Output matrix rexpanded
	 */
	public static MatrixBlock rexpand(MatrixBlock in, MatrixBlock ret, int max, boolean rows, boolean cast, boolean ignore, int k){
		//sanity check for input nnz (incl implicit handling of empty blocks)
		checkRexpand(in, ignore);

		//check for empty inputs (for ignore=true)
		if( in.isEmptyBlock(false) ) {
			if( rows )
				ret.reset(max, in.rlen, true);
			else //cols
				ret.reset(in.rlen, max, true);
			return ret;
		}

		//execute rexpand operations
		if( rows )
			return rexpandRows(in, ret, max, cast, ignore);
		else //cols
			return rexpandColumns(in, ret, max, cast, ignore, k);
	}


	/**
	 * The DML code to activate this function:
	 * <p>
	 * 
	 * ret = table(seq(1, nrow(A)), A, w)
	 * 
	 * @param seqHeight A sequence vector height.
	 * @param A         The MatrixBlock vector to encode.
	 * @param w         The weight matrix to multiply on output cells.
	 * @return A new MatrixBlock with the table result.
	 */
	public static MatrixBlock fusedSeqRexpand(int seqHeight, MatrixBlock A, double w) {
		return fusedSeqRexpand(seqHeight, A, w, null, true, 1);
	}

	/**
	 * The DML code to activate this function:
	 * <p>
	 * 
	 * ret = table(seq(1, nrow(A)), A, w)
	 * 
	 * @param seqHeight  A sequence vector height.
	 * @param A          The MatrixBlock vector to encode.
	 * @param w          The weight scalar to multiply on output cells.
	 * @param ret        The output MatrixBlock, does not have to be used, but depending on updateClen determine the
	 *                   output size.
	 * @param updateClen Update clen, if set to true, ignore dimensions of ret, otherwise use the column dimension of
	 *                   ret.
	 * @return A new MatrixBlock or ret.
	 */
	public static MatrixBlock fusedSeqRexpand(int seqHeight, MatrixBlock A, double w, MatrixBlock ret,
		boolean updateClen) {
		return fusedSeqRexpand(seqHeight, A, w, ret, updateClen, 1);
	}

	/**
	 * The DML code to activate this function:
	 * <p>
	 * 
	 * ret = table(seq(1, nrow(A)), A, w)
	 * 
	 * @param seqHeight  A sequence vector height.
	 * @param A          The MatrixBlock vector to encode.
	 * @param w          The weight matrix to multiply on output cells.
	 * @param ret        The output MatrixBlock, does not have to be used, but depending on updateClen determine the
	 *                   output size.
	 * @param updateClen Update clen, if set to true, ignore dimensions of ret, otherwise use the column dimension of
	 *                   ret.
	 * @param k			   Parallelization degree
	 * @return A new MatrixBlock or ret.
	 */
	public static MatrixBlock fusedSeqRexpand(int seqHeight, MatrixBlock A, double w, MatrixBlock ret,
		boolean updateClen, int k) {

		if(A.getNumRows() != seqHeight)
			throw new DMLRuntimeException(
				"Invalid input sizes for table \"table(seq(1, nrow(A)), A, w)\" : sequence height is: " + seqHeight
					+ " while A is: " + A.getNumRows());

		if(A.getNumColumns() > 1)
			throw new DMLRuntimeException(
				"Invalid input A in table(seq(1, nrow(A)), A, w): A should only have one column but has: "
					+ A.getNumColumns());

		if(!Double.isNaN(w) && w != 0) {
			if((CLALibRexpand.compressedTableSeq() || A instanceof CompressedMatrixBlock) && w == 1)
				return CLALibRexpand.rexpand(seqHeight, A, updateClen ? -1 : ret.getNumColumns(), k);
			else{
				return fusedSeqRexpandSparse(seqHeight, A, w, ret, updateClen);
			}
		}
		else {
			if(ret == null) {
				ret = new MatrixBlock();
				updateClen = true;
			}

			ret.rlen = seqHeight;
			// empty output.
			ret.denseBlock = null;
			ret.sparseBlock = null;
			ret.sparse = true;
			ret.nonZeros = 0;
			updateClenRexpand(ret, 0, updateClen);
			return ret;
		}

	}

	private static MatrixBlock fusedSeqRexpandSparse(int seqHeight, MatrixBlock A, double w, MatrixBlock ret,
													 boolean updateClen) {
		if(ret == null) {
			ret = new MatrixBlock();
			updateClen = true;
		}
		int outCols = updateClen ? -1 : ret.getNumColumns();
		final int rlen = seqHeight;
		// prepare allocation of CSR sparse block
		final int[] rowPointers = new int[rlen + 1];
		final int[] indexes = new int[rlen];
		final double[] values = new double[rlen];

		ret.rlen = rlen;
		// assign the output
		ret.sparse = true;
		ret.denseBlock = null;
		// construct sparse CSR block from filled arrays
		SparseBlockCSR csr = new SparseBlockCSR(rowPointers, indexes, values, seqHeight);
		ret.sparseBlock = csr;
		int blkz = Math.min(1024, seqHeight);
		int maxcol = 0;
		boolean containsNull = false;
		for(int i = 0; i < seqHeight; i += blkz) {
			// blocked execution for earlier JIT compilation
			int t = fusedSeqRexpandSparseBlock(csr, A, w, i, Math.min(i + blkz, seqHeight), updateClen,outCols);
			if(t < 0) {
				t = Math.abs(t);
				containsNull = true;
			}
			maxcol = Math.max(t, maxcol);
		}

		if(containsNull)
			csr.compact();

		rowPointers[seqHeight] = seqHeight;
		ret.setNonZeros(ret.sparseBlock.size());
		if(updateClen)
			ret.setNumColumns(outCols == -1 ? maxcol : (int) outCols);
		return ret;
	}

	private static int fusedSeqRexpandSparseBlock(final SparseBlockCSR csr, final MatrixBlock A, final double w, int rl,
												  int ru, boolean updateClen,int maxOutCol) {

		// prepare allocation of CSR sparse block
		final int[] rowPointers = csr.rowPointers();
		final int[] indexes = csr.indexes();
		final double[] values = csr.values();

		boolean containsNull = false;
		int maxCol = 0;

		for(int i = rl; i < ru; i++) {
			int c = rexpandSingleRow(i, A.get(i, 0), w, indexes, values, updateClen, maxOutCol);
			containsNull |= c < 0;
			maxCol = Math.max(c, maxCol);
			rowPointers[i] = i;
		}
	
		return containsNull ? -maxCol: maxCol;
	}

	private static void updateClenRexpand(MatrixBlock ret, int maxCol, boolean updateClen) {
		// update meta data (initially unknown number of columns)
		// Only allowed if we enable the update flag.
		if(updateClen)
			ret.clen = maxCol;
	}

	public static int rexpandSingleRow(int row, double v2, double w,  int[] retIx, double[] retVals,
												boolean updateClen, int maxOutCol) {

		final int colUnsafe = UtilFunctions.toInt(v2);	// colUnsafe = 0 for Nan
		int isNan = (Double.isNaN(v2) ? 1 : 0);		// avoid branching by boolean to int conversion
		int col = colUnsafe - isNan;				// col = -1 for Nan

		// use branch prediction for rare case
		if(!Double.isNaN(v2) && colUnsafe <= 0)
			throw new DMLRuntimeException("Erroneous input while computing the contingency table (value <= zero): " + v2);

		// avoid branching again by boolean to int conversion
		int valid = !Double.isNaN(v2) && (updateClen || col <= maxOutCol) ? 1 : 0;
		retIx[row] = (col - 1)*valid;		// use valid as switch
		retVals[row] = w*valid;
		return valid*col + valid - 1;		// -1 if invalid else col
	}

	/**
	 * Quick check if the input is valid for rexpand, this check does not guarantee that the input is valid for rexpand
	 * 
	 * @param in     Input matrix block
	 * @param ignore If zero valued cells should be ignored
	 */
	public static void checkRexpand(MatrixBlock in, boolean ignore){
		if( !ignore && in.getNonZeros() < in.getNumRows() )
			throw new DMLRuntimeException("Invalid input w/ zeros for rexpand ignore=false "
					+ "(rlen="+in.getNumRows()+", nnz="+in.getNonZeros()+").");
	}

	/**
	 * MR/Spark rexpand operation (single input, multiple outputs incl empty blocks)
	 * 
	 * @param data    Input indexed matrix block
	 * @param max     Total nrows/cols of the output
	 * @param rows    If the expansion is in rows direction
	 * @param cast    If the values contained should be cast to double (rounded up and down)
	 * @param ignore  Ignore if the input contain values below zero that technically is incorrect input.
	 * @param blen    The block size to slice the output up into
	 * @param outList The output indexedMatrixValues (a list to add all the output blocks to / modify)
	 */
	public static void rexpand(IndexedMatrixValue data, double max, boolean rows, boolean cast, boolean ignore, long blen, ArrayList<IndexedMatrixValue> outList) {
		//prepare parameters
		MatrixIndexes ix = data.getIndexes();
		MatrixBlock in = (MatrixBlock)data.getValue();
		
		//execute rexpand operations incl sanity checks
		//TODO more robust (memory efficient) implementation w/o tmp block
		MatrixBlock tmp = rexpand(in, new MatrixBlock(), max, rows, cast, ignore, 1);
		
		//prepare outputs blocks (slice tmp block into output blocks ) 
		if( rows ) //expanded vertically
		{
			for( int rl=0; rl<tmp.getNumRows(); rl+=blen ) {
				MatrixBlock mb = tmp.slice(
						rl, (int)(Math.min(rl+blen, tmp.getNumRows())-1));
				outList.add(new IndexedMatrixValue(
						new MatrixIndexes(rl/blen+1, ix.getRowIndex()), mb));
			}
		}
		else //expanded horizontally
		{
			for( int cl=0; cl<tmp.getNumColumns(); cl+=blen ) {
				MatrixBlock mb = tmp.slice(
						0, tmp.getNumRows()-1,
						cl, (int)(Math.min(cl+blen, tmp.getNumColumns())-1), new MatrixBlock());
				outList.add(new IndexedMatrixValue(
						new MatrixIndexes(ix.getRowIndex(), cl/blen+1), mb));
			}
		}
	}

	///////////////////////////////
	// private CP implementation //
	///////////////////////////////

	private static ReorgType getReorgType( ReorgOperator op )
	{
		if( op.fn instanceof SwapIndex )  //transpose
			return ReorgType.TRANSPOSE;
		
		else if( op.fn instanceof RevIndex ) //rev
			return ReorgType.REV;

		else if( op.fn instanceof RollIndex ) //roll
			return ReorgType.ROLL;

		else if( op.fn instanceof DiagIndex ) //diag
			return ReorgType.DIAG;
		
		else if( op.fn instanceof SortIndex ) //sort
			return ReorgType.SORT;
				
		return ReorgType.INVALID;
	}

	private static void transposeDenseToDense(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu) {
		final int m = in.rlen;
		final int n = in.clen;
		final int n2 = out.clen;
		
		DenseBlock a = in.getDenseBlock();
		DenseBlock c = out.getDenseBlock();
		
		if( m==1 || n==1 ) //VECTOR TRANSPOSE
		{
			//plain memcopy, in case shallow dense copy no applied
			//input/output guaranteed single block
			int ix = rl+cl; int len = ru+cu-ix-1;
			System.arraycopy(a.valuesAt(0), ix, c.valuesAt(0), ix, len);
		}
		else //MATRIX TRANSPOSE
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128;
			
			//blocked execution
			if( a.numBlocks()==1 && c.numBlocks()==1 ) { //<16GB
				double[] avals = a.valuesAt(0);
				double[] cvals = c.valuesAt(0);
				for( int bi = rl; bi<ru; bi+=blocksizeI ) {
					int bimin = Math.min(bi+blocksizeI, ru);
					for( int bj = cl; bj<cu; bj+=blocksizeJ ) {
						int bjmin = Math.min(bj+blocksizeJ, cu);
						//core transpose operation
						for( int i=bi; i<bimin; i++ ) {
							int aix = i * n + bj;
							int cix = bj * n2 + i;
							transposeRow(avals, cvals, aix, cix, n2, bjmin-bj);
						}
					}
				}
			}
			else { //general case > 16GB (multiple blocks)
				for( int bi = rl; bi<ru; bi+=blocksizeI ) {
					int bimin = Math.min(bi+blocksizeI, ru);
					for( int bj = cl; bj<cu; bj+=blocksizeJ ) {
						int bjmin = Math.min(bj+blocksizeJ, cu);
						//core transpose operation
						for( int i=bi; i<bimin; i++ ) {
							double[] avals = a.values(i);
							int aix = a.pos(i);
							for( int j=bj; j<bjmin; j++ )
								c.set(j, i, avals[ aix+j ]);
						}
					}
				}
			}
		}
	}

	private static void transposeDenseToSparse(MatrixBlock in, MatrixBlock out){
		transposeDenseToSparse(in, out, 1);
	}

	private static void transposeDenseToSparse(MatrixBlock in, MatrixBlock out, int k){
		if( out.rlen == 1 ) 
			transposeDenseToSparseVV(in,out);
		else
			transposeDenseToSparseMM(in, out, k);
	}

	private static void transposeDenseToSparseVV(MatrixBlock in, MatrixBlock out){
		final int m = in.rlen;
		final DenseBlock a = in.getDenseBlock();
		out.allocateSparseRowsBlock(false);
		final SparseBlock c = out.getSparseBlock();
		c.set(0, new SparseRowVector((int)in.nonZeros, a.valuesAt(0), m), false);
	}

	private static void transposeDenseToSparseMM(MatrixBlock in, MatrixBlock out, int k){
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = (int) (in.nonZeros/m2); 
		final DenseBlock a = in.getDenseBlock();

		final SparseRowVector[] rows = new SparseRowVector[m2];
		for(int j = 0; j < m2; j++)
			rows[j] = new SparseRowVector(ennz2, n2);

		if(k <= 1)
			transposeDenseToSparseMMRange(a, rows, 0, m, 0, n);
		else {
			final ExecutorService pool = CommonThreadPool.get(k);
			try {
				final ArrayList<TransposeDenseToSparseTask> tasks = new ArrayList<>();
				final int rbz = Math.max(1, m2 / k);
				for(int i = 0; i < m2; i += rbz)
					tasks.add(new TransposeDenseToSparseTask(a, rows, 0, m, i, Math.min(i + rbz, n)));
				for(Future<Object> task : pool.invokeAll(tasks))
					task.get();
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}

		SparseBlock c = new SparseBlockMCSR(rows, false);
		out.setSparseBlock(c);
	}

	private static void transposeDenseToSparseMMRange(DenseBlock a, SparseRowVector[] rows, int rl, int ru, int cl,
		int cu) {
		// blocking according to typical L2 cache sizes
		final int blocksizeI = 128;
		final int blocksizeJ = 128;
		for(int bi = rl; bi < ru; bi += blocksizeI) {
			final int bimin = Math.min(bi + blocksizeI, ru);
			for(int bj = cl; bj < cu; bj += blocksizeJ) {
				final int bjmin = Math.min(bj + blocksizeJ, cu);
				// core transpose operation
				for(int i = bi; i < bimin; i++) {
					final double[] avals = a.values(i);
					final int aix = a.pos(i);
					for(int j = bj; j < bjmin; j++)
						rows[j].append(i, avals[aix + j]);
				}
			}
		}
	}

	private static class TransposeDenseToSparseTask implements Callable<Object> {
		private DenseBlock a;
		private SparseRowVector[] rows;
		private int rl;
		private int ru;
		private int cl;
		private int cu;

		protected TransposeDenseToSparseTask(DenseBlock a, SparseRowVector[] rows, int rl, int ru, int cl, int cu) {
			this.a = a;
			this.rows = rows;
			this.rl = rl;
			this.ru = ru;
			this.cl = cl;
			this.cu = cu;
		}

		@Override
		public Object call() {
			transposeDenseToSparseMMRange(a, rows, rl, ru, cl, cu);
			return null;
		}
	}

	private static void transposeUltraSparse(MatrixBlock in, MatrixBlock out) {
		//note: applied if nnz < max(rlen, clen) - so no cache blocking
		// but basic, naive transposition in a single-threaded context
		Iterator<IJV> iter = in.getSparseBlockIterator();
		SparseBlock b = out.getSparseBlock();
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			b.append(cell.getJ(), cell.getI(), cell.getV());
		}
		out.setNonZeros(in.getNonZeros());
	}
	
	private static void transposeUltraSparse(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu) {
		Iterator<IJV> iter = in.getSparseBlockIterator(rl, ru, cl, cu);
		SparseBlock b = out.getSparseBlock();
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			b.append(cell.getJ(), cell.getI(), cell.getV());
		}
	}
	
	private static void transposeSparseToSparse(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu,
		int[] cnt) {
		// NOTE: called only in sequential or column-wise parallel execution
		if(rl > 0 || ru < in.rlen)
			throw new RuntimeException("Unsupported row-parallel transposeSparseToSparse: " + rl + ", " + ru);
		else if(cu - cl == 1) // SINGLE TARGET ROW
			transposeSparseToSparseRow(in, out, rl, ru, cl, cnt);
		else
			transposeSparseToSparseBlock(in, out, rl, ru, cl, cu, cnt);
	}

	private static void transposeSparseToSparseRow(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int[] cnt){
		
		final SparseBlock a = in.getSparseBlock();
		final SparseBlock c = out.getSparseBlock();

		//number of columns <= num cores, use sequential scan over input
		//and avoid cache blocking and temporary position maintenance
		if( cnt[cl] > 0 )
			c.allocate(cl, cnt[cl]);
		
		for( int i=rl; i<ru; i++ ) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			for( int j=apos; j<apos+alen && aix[j]<=cl; j++ )
				if( aix[j] == cl )
					c.append(cl, i, avals[j]);
		}
	}


	private static void transposeSparseToSparseBlock(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu, int[] cnt){
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = (int) (in.nonZeros / m2); 
		
		final SparseBlock a = in.getSparseBlock();
		final SparseBlock c = out.getSparseBlock();

		// allocate output sparse rows
		if(cnt != null){
			for(int i = cl; i < cu; i++){
				if(cnt[i] > 0)
					c.allocate(i, cnt[i]);
			}
		}
		else {
			for(int i = cl; i < cu; i++)
				c.allocate(i, Math.max(ennz2, 2), n2);
		}
		
		//blocking according to typical L2 cache sizes w/ awareness of sparsity
		final long xsp = (long)in.rlen * in.clen / in.nonZeros;
		final int blocksizeI = Math.max(128, (int) (8*xsp));
		final int blocksizeJ = Math.max(128, (int) (8*xsp));

		if(blocksizeJ * 2 > m2 && c instanceof SparseBlockMCSR)
			transposeSparseToSparseBlockTallSkinny(a, (SparseBlockMCSR)c, blocksizeI, rl, ru, cl, cu);
		else if(c instanceof SparseBlockMCSR)
			transposeSparseToSparseBlockMCSR(a, (SparseBlockMCSR) c, blocksizeI, blocksizeJ, rl, ru, cl, cu);
		else 
			transposeSparseToSparseBlockGeneric(a, c, blocksizeI, blocksizeJ, rl, ru, cl, cu);
		
	}

	private static void transposeSparseToSparseBlockTallSkinny(final SparseBlock a, final SparseBlockMCSR c,
		final int blocksizeI, final int rl, final int ru, final int cl, final int cu) {

		final SparseRow[] sr = c.getRows();
		for(int i = rl; i < ru; i++) {
			if(a.isEmpty(i))
				continue;
			int j = a.posFIndexGTE(i, cl); // last block boundary
			if(j >= 0) {
				final int apos = a.pos(i);
				final int alen = a.size(i);
				final int[] aix = a.indexes(i);
				final double[] avals = a.values(i);
				for(j = j + apos; j < apos + alen && aix[j] < cu; j++) {
					sr[aix[j]].append(i, avals[j]);
				}
			}
		}
	}


	private static void transposeSparseToSparseBlockMCSR(SparseBlock a, SparseBlockMCSR c, final int blocksizeI,
		final int blocksizeJ, int rl, int ru, int cl, int cu) {
		// temporary array for block boundaries (for preventing binary search)
		final int[] ix = new int[Math.min(blocksizeI, ru - rl)];

		final SparseRow[] sr = c.getRows();
		// blocked execution
		for(int bi = rl; bi < ru; bi += blocksizeI) {
			Arrays.fill(ix, 0);
			// find column starting positions
			int bimin = Math.min(bi + blocksizeI, ru);
			if(cl > 0) {
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					int j = a.posFIndexGTE(i, cl);
					ix[i - bi] = (j >= 0) ? j : a.size(i);
				}
			}

			for(int bj = cl; bj < cu; bj += blocksizeJ) {
				int bjmin = Math.min(bj + blocksizeJ, cu);
				// core block transpose operation
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					final int apos = a.pos(i);
					final int alen = a.size(i);
					final int[] aix = a.indexes(i);
					final double[] avals = a.values(i);
					int j = ix[i - bi] + apos; // last block boundary
					for(; j < apos + alen && aix[j] < bjmin; j++)
						sr[aix[j]] = sr[aix[j]].append(i, avals[j]);
					ix[i - bi] = j - apos; // keep block boundary
				}
			}
		}
	}

	private static void transposeSparseToSparseBlockGeneric(SparseBlock a, SparseBlock c, final int blocksizeI,
		final int blocksizeJ, int rl, int ru, int cl, int cu) {
		// temporary array for block boundaries (for preventing binary search)
		final int[] ix = new int[Math.min(blocksizeI, ru - rl)];

		// blocked execution
		for(int bi = rl; bi < ru; bi += blocksizeI) {
			Arrays.fill(ix, 0);
			// find column starting positions
			int bimin = Math.min(bi + blocksizeI, ru);
			if(cl > 0) {
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					int j = a.posFIndexGTE(i, cl);
					ix[i - bi] = (j >= 0) ? j : a.size(i);
				}
			}

			for(int bj = cl; bj < cu; bj += blocksizeJ) {
				int bjmin = Math.min(bj + blocksizeJ, cu);
				// core block transpose operation
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					final int apos = a.pos(i);
					final int alen = a.size(i);
					final int[] aix = a.indexes(i);
					final double[] avals = a.values(i);
					int j = ix[i - bi] + apos; // last block boundary
					for(; j < apos + alen && aix[j] < bjmin; j++)
						c.append(aix[j], i, avals[j]);

					ix[i - bi] = j - apos; // keep block boundary
				}
			}
		}
	}

	private static void transposeSparseToSparseCSR(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu,
		int[] cnt) {
		// NOTE: called only in sequential or column-wise parallel execution
		if(rl > 0 || ru < in.rlen)
			throw new RuntimeException("Unsupported row-parallel transposeSparseToSparse: " + rl + ", " + ru);
		if(cu - cl == 1)
			transposeSparseToSparseCSRSingleCol(in, out, rl, ru, cl, cu, cnt);
		else if(in.getSparseBlock() instanceof SparseBlockCSR)
			transposeSparseCSRToSparseCSRMultiCol(in, out, cl, cu, cnt);
		else
			transposeSparseToSparseCSRMultiCol(in, out, cl, cu, cnt);
	}

	private final static void transposeSparseCSRToSparseCSRMultiCol(final MatrixBlock in, final MatrixBlock out,
		final int cl, final int cu, final int[] cnt) {
		final int rlen = in.rlen;

		final SparseBlockCSR a = (SparseBlockCSR) in.getSparseBlock();
		final SparseBlockCSR c = (SparseBlockCSR) out.getSparseBlock();

		final long xsp = (long) rlen * in.clen / in.nonZeros;
		final int blocksizeI = Math.min(Math.max(128, (int) (8 * xsp)), 512);

		// temporary array for block boundaries (for preventing binary search)
		final int[] ix = new int[Math.min(blocksizeI, rlen)];

		// blocked execution
		for(int bi = 0; bi < rlen; bi += blocksizeI)
			transposeSparseCSRToSparseCSRMultiColBlock(bi, blocksizeI, rlen, cl, cu, ix, a, cnt, c);
		
	}

	private final static void transposeSparseCSRToSparseCSRMultiColBlock(int bi, int blocksizeI, int rlen, int cl,
		int cu, int[] ix, SparseBlockCSR a, final int[] cnt, SparseBlockCSR c) {

		final int[] aix = a.indexes();
		final double[] avals = a.values();
		final int[] outIndexes = c.indexes();
		final double[] outValues = c.values();

		// find column starting positions
		final int bimin = Math.min(bi + blocksizeI, rlen);
		if(cl > 0)
			fillSkip(bi, bimin, a, cl, ix);
		else 
			Arrays.fill(ix, 0);

		for(int bj = cl; bj < cu; bj += blocksizeI) 
			transposeSparseCSRToSparseCSRMultiColBlockBlock(bi, bj, bimin, cu, blocksizeI, a, ix, aix, avals, outIndexes,
				outValues, cnt);
	}

	private final static void transposeSparseCSRToSparseCSRMultiColBlockBlock(int bi, int bj, int bimin, int cu,
		int blocksizeI, SparseBlockCSR a, int[] ix, int[] aix, double[] avals, int[] outIndexes, double[] outValues, int[] cnt) {
		final int bjmin = Math.min(bj + blocksizeI, cu);
		// core block transpose operation
		for(int i = bi; i < bimin; i++) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			int j = ix[i - bi] + apos; // last block boundary
			for(; j < apos + alen && aix[j] < bjmin; j++) {
				int pointer = cnt[aix[j]];
				cnt[aix[j]]++;
				outIndexes[pointer] = i;
				outValues[pointer] = avals[j];
			}
			ix[i - bi] = j - apos; // keep block boundary
		}
	}

	private final static void fillSkip(int bi, int bimin, SparseBlockCSR a, int cl, int[] ix){
		// fill the skip boundaries.
		for(int i = bi; i < bimin; i++) {
				int j = a.posFIndexGTE(i, cl);
				ix[i - bi] = (j >= 0) ? j : a.size(i);
			}
	}

	private final static void transposeSparseToSparseCSRMultiCol(final MatrixBlock in, final MatrixBlock out,
		final int cl, final int cu, final int[] cnt) {
		final int rlen = in.rlen;

		final SparseBlock a = in.getSparseBlock();
		final SparseBlockCSR c = (SparseBlockCSR) out.getSparseBlock();

		final int[] outIndexes = c.indexes();
		final double[] outValues = c.values();

		final long xsp = (long) rlen * in.clen / in.nonZeros;
		final int blocksizeI = Math.min(Math.max(128, (int) (8 * xsp)), 512);

		// temporary array for block boundaries (for preventing binary search)
		final int[] ix = new int[Math.min(blocksizeI, rlen)];

		// blocked execution
		for(int bi = 0; bi < rlen; bi += blocksizeI) {
			Arrays.fill(ix, 0);
			// find column starting positions
			int bimin = Math.min(bi + blocksizeI, rlen);
			if(cl > 0) {
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					int j = a.posFIndexGTE(i, cl);
					ix[i - bi] = (j >= 0) ? j : a.size(i);
				}
			}

			for(int bj = cl; bj < cu; bj += blocksizeI) {
				final int bjmin = Math.min(bj + blocksizeI, cu);
				// core block transpose operation
				for(int i = bi; i < bimin; i++) {
					if(a.isEmpty(i))
						continue;
					final int apos = a.pos(i);
					final int alen = a.size(i);
					final int[] aix = a.indexes(i);
					final double[] avals = a.values(i);
					int j = ix[i - bi] + apos; // last block boundary
					for(; j < apos + alen && aix[j] < bjmin; j++) {
						int pointer = cnt[aix[j]];
						cnt[aix[j]]++;
						outIndexes[pointer] = i;
						outValues[pointer] = avals[j];
					}
					ix[i - bi] = j - apos; // keep block boundary
				}
			}
		}
	}

	private static void transposeSparseToSparseCSRSingleCol(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl,
		int cu, int[] cnt) {
		final SparseBlock a = in.getSparseBlock();
		final SparseBlockCSR c = (SparseBlockCSR) out.getSparseBlock();

		final int[] outIndexes = c.indexes();
		final double[] outValues = c.values();

		int i = 0;
		final int end = c.size(cl) + c.pos(cl);
		int outPointer = cnt[cl];
		while(outPointer < end) {
			if(!a.isEmpty(i)) {
				final int apos = a.pos(i);
				final int alen = a.size(i);
				final int[] aix = a.indexes(i);
				final double[] avals = a.values(i);
				for(int j = apos; j < apos + alen && aix[j] <= cl; j++)
					if(aix[j] == cl) {
						outIndexes[outPointer] = i;
						outValues[outPointer] = avals[j];
						outPointer++;
					}
			}
			i++;
		}

	}

	private static void transposeSparseToDense(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu) {
		final int m = in.rlen;
		final int n = in.clen;
		
		SparseBlock a = in.getSparseBlock();
		DenseBlock c = out.getDenseBlock();
		
		if( m==1 ) //ROW VECTOR TRANSPOSE
		{
			//NOTE: called only in sequential execution
			int alen = a.size(0); //always pos 0
			int[] aix = a.indexes(0);
			double[] avals = a.values(0);
			double[] cvals = c.valuesAt(0);
			for( int j=0; j<alen; j++ )
				cvals[aix[j]] = avals[j];
		}
		else //MATRIX TRANSPOSE
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128; 
		
			//temporary array for block boundaries (for preventing binary search) 
			int[] ix = new int[blocksizeI];
			
			//blocked execution
			for( int bi = rl; bi<ru; bi+=blocksizeI ) {
				Arrays.fill(ix, 0);
				int bimin = Math.min(bi+blocksizeI, ru);
				for( int bj = 0; bj<n; bj+=blocksizeJ ) {
					int bjmin = Math.min(bj+blocksizeJ, n);
					//core transpose operation
					for( int i=bi, iix=0; i<bimin; i++, iix++ ) {
						if( a.isEmpty(i) ) continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						int j = ix[iix]; //last block boundary
						for( ; j<alen && aix[apos+j]<bjmin; j++ )
							c.set(aix[apos+j], i, avals[apos+j]);
						ix[iix] = j; //keep block boundary
					}
				}
			}
		}
	}
	
	/**
	 * Using C2R & R2C algorithm from PPOP 2014.
	 * 
	 * https://dl.acm.org/doi/pdf/10.1145/2692916.2555253
	 * 
	 * @param in The matrix to transpose in place.
	 * @param k  The number of threads allowed to be used.
	 */
	private static void transposeInPlaceDense(MatrixBlock in, int k){
		DenseBlock values = in.getDenseBlock();
		if(values.numBlocks()>1)
			throw new NotImplementedException("Not Implemented in place transpose with more than one block");
		
		// Swap rows and cols
		final int cols = in.getNumRows();
		final int rows = in.getNumColumns();
		
		if(cols == 1 || rows == 1){
			values.setDims(new int[] {rows, cols});
			in.setNumColumns(cols);
			in.setNumRows(rows);
			// swap rows and column numbers;
		}
		else if(cols == rows){
			// If the number of rows equals the number of columns simply swap each element along the diagonal.
			// This only results in half - number of diagonal elements swaps.
			transposeInPlaceTrivial(in.getDenseBlockValues(), cols, k);
		}
		else {
			if(TRANSPOSE_IN_PLACE_DENSE_LEGACY) {
				if(cols < rows) {
					// important to set dims after
					c2r(in, k);
					values.setDims(new int[] {rows, cols});
					in.setNumColumns(cols);
					in.setNumRows(rows);
				}
				else {
					// important to set dims before
					values.setDims(new int[] {rows, cols});
					in.setNumColumns(cols);
					in.setNumRows(rows);
					r2c(in, k);
				}
			} else {
				transposeInPlaceDenseBrenner(in, k);
			}
		}

		
	}

	/** Thread local temporary double array.. */
	private static ThreadLocal<double[]> memPool = new ThreadLocal<>() {
		@Override
		protected double[] initialValue() {
			return null;
		}
	};

	/**
	 * Only use if the number of rows and cols are equal
	 * @param values The values in the dense matrix.
	 * @param rowAndCols The number of rows & cols.
	 * @param k The number of threads allowed to be used.
	 */
	private static void transposeInPlaceTrivial(double[] values, int rowAndCols, int k){
		if(rowAndCols > 15){
			ExecutorService pool = CommonThreadPool.get(k);
			try{
				ArrayList<TransposeInPlaceTrivialTask> tasks = new ArrayList<>();
				int blklen = 128;
				for(int i = 0; i * blklen < rowAndCols; i++){
					for(int j = i; j * blklen < rowAndCols; j++){
						tasks.add(new TransposeInPlaceTrivialTask(
							i * blklen, Math.min((i+1) * blklen,rowAndCols),
							j * blklen, Math.min((j+1) * blklen,rowAndCols), rowAndCols, values));
					}
				}
	
				List<Future<Object>> rtasks = pool.invokeAll(tasks);

				for(Future<Object> rt : rtasks)
					rt.get();
			}
			catch(InterruptedException | ExecutionException ex) {
				throw new DMLRuntimeException("Failed parallel transpose in place with equal number col and rows.", ex);
			}
			finally{
				pool.shutdown();
			}
		}else{
			for(int rowidx = 0; rowidx < rowAndCols; rowidx++){
				for(int colidx = rowidx+1; colidx < rowAndCols; colidx++){
					swap(values, rowidx * rowAndCols + colidx, colidx * rowAndCols + rowidx);
				}
			}
		}
	}

	private static class TransposeInPlaceTrivialTask implements Callable<Object>{
		private final int _rowStart;
		private final int _rowStop;
		private final int _colStart;
		private final int _colStop;
		private final int _rowAndCols;
		private final double[] _values;

		TransposeInPlaceTrivialTask(int rowStart, int rowStop, int colStart, int colStop, int rowAndCols, double[] values){
			_rowStart = rowStart;
			_rowStop = rowStop;
			_colStart = colStart;
			_colStop = colStop;
			_rowAndCols = rowAndCols;
			_values = values;
		}

		@Override
		public Object call() {
			for(int rowidx = _rowStart; rowidx < _rowStop; rowidx++){
				for(int colidx = Math.max(rowidx+1, _colStart); colidx < _colStop; colidx++){
					swap(_values, rowidx * _rowAndCols + colidx, colidx * _rowAndCols + rowidx);
				}
			}
			return null;
		}
	}

	private static void swap(double[] values, int from, int to){
		double tmp = values[from];
		values[from] = values[to];
		values[to] = tmp;
	}

	private static void c2r(MatrixBlock in, int k){
		double[] A = in.getDenseBlockValues();
		int m = in.getNumRows();
		int n = in.getNumColumns();
		int c = gcd(m,n);
		int a = m/c;
		int b = n/c;

		double[] tmp = memPool.get();
		if(tmp == null) {
			memPool.set(new double[Math.max(m, n)]);
			tmp = memPool.get();
		}

		ExecutorService pool = CommonThreadPool.get(k);
		
		// Column rotate Gather
		try {
			ArrayList<Callable<Object>> tasks = new ArrayList<>();
			if(c > 1) {
				if(m > 10 && n > 100) {
					int blkz = Math.max((n - c) / k, 1);
					for(int j = c; j * blkz < n; j++) {
						tasks.add(new rTask(A, j * blkz, Math.min((j + 1) * blkz, n), b, n, m));
					}
					for(Future<Object> rt : pool.invokeAll(tasks))
						rt.get();
					tasks.clear();

				}
				else {
					for(int j = c; j < n; j++) {
						rj(tmp, A, j, b, n, m);
					}
				}
			}

			// Row shuffle Scatter
			if(m > 10 && n > 100) {
				int blkz = Math.max(m / k, 1);
				for(int i = 0; i * blkz < m; i++) {
					tasks.add(new dTask(A, i * blkz, Math.min((i + 1) * blkz, m), b, n, m));
				}
				for(Future<Object> rt : pool.invokeAll(tasks))
					rt.get();
				tasks.clear();

			}
			else {
				for(int i = 0; i < m; i++) {
					di(tmp, A, i, b, n, m);
				}
			}

			// Column shuffle Gather
			if(m > 10 && n > 100) {
				int blkz = Math.max(n / k, 1);
				for(int j = 0; j * blkz < n; j++) {
					tasks.add(new sTask(A, j * blkz, Math.min((j + 1) * blkz, n), a, n, m));
				}
				for(Future<Object> rt : pool.invokeAll(tasks))
					rt.get();
				tasks.clear();

			}
			else {
				for(int j = 0; j < n; j++) {
					sj(tmp, A, j, a, n, m);
				}
			}
			memPool.remove();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException("Failed parallel c2r transpose in column rotate step", ex);
		}
		finally{
			pool.shutdown();
		}
	}

	private static void rj(double[] tmp, double[] A, int j, int b, int n, int m){
		int part = j / b;
		for (int i = 0; i< m; i++){
			int rj = (i+part) % m;
			tmp[i] = A[rj*n + j];
		}
		
		for (int i = j, off = 0; i< m*n; i+= n, off++){
			A[i] = tmp[off];
		}
	}

	private static class rTask implements Callable<Object> {
		final double[] _A;

		final int _jStart;
		final int _jEnd;
		final int _b;
		final int _n;
		final int _m;

		rTask(double[] A, int jStart, int jEnd, int b, int n, int m){
			_A = A;
			_jStart = jStart;
			_jEnd = jEnd;
			_b = b;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}
			for( int j = _jStart; j< _jEnd; j++){

				rj(tmp, _A, j, _b, _n, _m);
			}
			return null;
		}
	}

	private static void di(double[] tmp, double[] A, int i, int b, int n, int m){
		int off = i*n;
		for(int j = 0; j< n; j++, off++){
			int dij =((i + j/b) % m + j*m) % n;
			tmp[dij] = A[off];
		}
		
		System.arraycopy(tmp, 0, A, i*n, n);

	}

	private static class dTask implements Callable<Object>{
		final double[] _A;

		final int _iStart;
		final int _iEnd;
		final int _b;
		final int _n;
		final int _m;

		dTask(double[] A, int iStart, int iEnd, int b, int n, int m){
			_A = A;
			_iStart = iStart;
			_iEnd = iEnd;
			_b = b;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}

			for (int i = _iStart; i< _iEnd; i++){

				di(tmp, _A, i, _b, _n, _m);
			}
			return null;
		}
	}

	private static void sj(double[] tmp, double[] A, int j, int a, int n, int m){

		for (int i = 0; i< m; i++){
			int sji = ((j + i*n - i/a) % m )* n;
			tmp[i] = A[sji + j];
		}

		for (int i = j, off = 0; i< m * n; i += n, off ++){
			A[i] = tmp[off];
			
		}
	}

	private static class sTask implements Callable<Object>{
		final double[] _A;

		// final int _j;
		final int _jStart;
		final int _jEnd;
		final int _a;
		final int _n;
		final int _m;

		sTask(double[] A, int jStart, int jEnd, int a, int n, int m){
			_A = A;
			_jStart = jStart;
			_jEnd = jEnd;
			_a = a;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}
			for( int j = _jStart; j< _jEnd; j++){

				sj(tmp, _A, j, _a, _n, _m);
			}
			return null;
		}
	}

	private static void r2c(MatrixBlock in, int k){
		double[] A = in.getDenseBlockValues();
		int m = in.getNumRows();
		int n = in.getNumColumns();
		int c = gcd(m,n);
		int a = m/c;
		int b = n/c;
		int a_inv = modInverse(a,b);

		double[] tmp = memPool.get();
		if(tmp == null) {
			memPool.set(new double[Math.max(m, n)]);
			tmp = memPool.get();
		}

		ExecutorService pool = CommonThreadPool.get(k);
		
		try {
			ArrayList<Callable<Object>> tasks = new ArrayList<>();
			if(m > 10 && n > 100) {
				int blkz = Math.max(n / k, 1);
				for(int j = 0; j * blkz < n; j++) {
					tasks.add(new s_invTask(A, j * blkz, Math.min((j + 1) * blkz, n), a, n, m));
				}
				for(Future<Object> rt : pool.invokeAll(tasks))
					rt.get();
				tasks.clear();

			}
			else {
				for(int j = 0; j < n; j++) {
					sj_inv(tmp, A, j, a, n, m);
				}
			}

			if(m > 10 && n > 100) {
				int blkz = Math.max(m / k, 1);
				for(int i = 0; i * blkz < m; i++) {
					tasks.add(new d_invTask(A, i * blkz, Math.min((i + 1) * blkz, m), a_inv, b, c, n, m));
				}
				for(Future<Object> rt : pool.invokeAll(tasks))
					rt.get();
				tasks.clear();

			}
			else {
				if(b * b < 0) {
					// if there is a risk for overflow use the safe method.
					for(int i = 0; i < m; i++)
						di_inv_safe(tmp, A, i, a_inv, b, c, n, m);
				}
				else {
					for(int i = 0; i < m; i++)
						di_inv(tmp, A, i, a_inv, b, c, n, m);
				}

			}

			if(c > 1) {
				if(m > 10 && n > 100) {
					int blkz = Math.max((n - c) / k, 1);
					for(int j = c; j * blkz < n; j++) {
						tasks.add(new r_invTask(A, j * blkz, Math.min((j + 1) * blkz, n), b, n, m));
					}
					for(Future<Object> rt : pool.invokeAll(tasks))
						rt.get();
					tasks.clear();

				}
				else {

					for(int j = c; j < n; j++) {
						rj_inv(tmp, A, j, b, n, m);
					}
				}
			}

			memPool.remove();
		}
		catch(InterruptedException | ExecutionException ex) {
			throw new DMLRuntimeException("Failed parallel r2c transpose in place in inverse colum shuffle.", ex);
		}
		finally {
			pool.shutdown();
		}
	}

	private static void sj_inv(double[] tmp, double[] A, int j, int a, int n, int m){
			// This deviate from the paper, since this implementation
			// is leveraging the location switch by not assigning into 
			// the temp array, in order, but based on which elements to shift.
			for (int i = 0, off = 0; i< m * n; i+= n, off++){
				int sji = ((j + i - (off/a)) % m );
				tmp[sji] = A[i + j];
			}
			for (int i = j, off = 0; i< m * n; i+= n, off++){
				A[i] = tmp[off];
			}
	}

	private static class s_invTask implements Callable<Object>{
		final double[] _A;

		// final int _j;
		final int _jStart;
		final int _jEnd;
		final int _a;
		final int _n;
		final int _m;

		s_invTask(double[] A, int jStart, int jEnd, int a, int n, int m){
			_A = A;
			_jStart = jStart;
			_jEnd = jEnd;
			_a = a;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}
			for( int j = _jStart; j< _jEnd; j++){

				sj_inv(tmp, _A, j, _a, _n, _m);
			}
			return null;
		}
	}

	private static void di_inv(double[] tmp, double[] A, int i, int a_inv, int b, int c, int n, int m){
		int off = i*n;
		final int tmpIC = i + c;
		final int tmpIN = i * (n - 1);
		for(int j = 0; j < n; j++, off++){
			int f = (tmpIC - (j % c ) <= m) ? j + tmpIN : j + tmpIN +  m;
			int dij_inverse = ((a_inv % b)  * ((f/c) % b)) % b  + (f % c) * b;

			tmp[dij_inverse] = A[off];
		}
	
		System.arraycopy(tmp, 0, A, i*n, n);
	}

	private static void di_inv_safe(double[] tmp, double[] A, int i, int a_inv, int b, int c, int n, int m){
		int off = i*n;
		final int tmpIC = i + c;
		final int tmpIN = i * (n - 1);
		for(int j = 0; j < n; j++, off++){
			int f = (tmpIC - (j % c ) <= m) ? j + tmpIN : j + tmpIN +  m;
			int dij_inverse = ((int)((((long)(a_inv % b) * ((f/c) % b)) % b)) + (f % c) * b);
			
			tmp[dij_inverse] = A[off];
		}

		System.arraycopy(tmp, 0, A, i*n, n);
	}

	private static class d_invTask implements Callable<Object>{
		final double[] _A;

		final int _iStart;
		final int _iEnd;
		final int _a_inv;
		final int _b;
		final int _c;
		final int _n;
		final int _m;

		d_invTask(double[] A, int iStart, int iEnd, int a_inv, int b, int c, int n, int m){
			_A = A;
			_iStart = iStart;
			_iEnd = iEnd;
			_a_inv = a_inv;
			_b = b;
			_c = c;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}
			// if( _b * _b < 0){
				// if there is a risk for overflow use the safe method.
				for (int i = _iStart; i< _iEnd; i++)
					di_inv_safe(tmp, _A, i,_a_inv, _b, _c, _n, _m);
			// }else{
			// 	for (int i = _iStart; i< _iEnd; i++)
			// 		di_inv(tmp, _A, i,_a_inv, _b, _c, _n, _m);
			// }
			return null;
		}
	}

	private static void rj_inv(double[] tmp, double[] A, int j, int b, int n, int m){
		int part = j/ b;
		for (int i = 0; i< m; i++){
			int rj = (i-part) % m;
			if(rj < 0)
				rj += m;
			tmp[i] = A[rj*n + j];
		}
		for (int i = j, off = 0; i< m * n; i+= n, off++){
			A[i] = tmp[off];
		}
	}

	private static class r_invTask implements Callable<Object>{
		final double[] _A;

		// final int _j;
		final int _jStart;
		final int _jEnd;
		final int _b;
		final int _n;
		final int _m;

		r_invTask(double[] A, int jStart, int jEnd, int b, int n, int m){
			_A = A;
			_jStart = jStart;
			_jEnd = jEnd;
			_b = b;
			_n = n;
			_m = m;
		}

		@Override
		public Object call(){
			double[] tmp = memPool.get();
			if(tmp == null) {
				memPool.set(new double[Math.max(_m,_n)]);
				tmp = memPool.get();
			}
			for( int j = _jStart; j< _jEnd; j++){
				rj_inv(tmp, _A, j, _b, _n, _m);
			}
			return null;
		}
	}

	private static int modInverse(int a , int m){
		a = a % m; 
		for(int x = 1; x < m; x++)
			if((a * x) % m == 1)
				return x;
		return 1;
		// TODO use optimized mod inverse operation.
		// This makes little differnece on overall algorithm performance
		// performance of algorithm.

	}

	private static int gcd(int a, int b){
		return a == 0 ? b : gcd(b % a, a);
	}

	static void transposeRow( double[] a, double[] c, int aix, int cix, int n2, int len ) {
		final int bn = len%8;
		//compute rest (not aligned to 8-blocks)
		for( int j=0; j<bn; j++, aix++, cix+=n2 )
			c[ cix ] = a[ aix+0 ];
		//unrolled 8-blocks
		for( int j=bn; j<len; j+=8, aix+=8, cix+=8*n2 ) {
			c[ cix + 0*n2 ] = a[ aix+0 ];
			c[ cix + 1*n2 ] = a[ aix+1 ];
			c[ cix + 2*n2 ] = a[ aix+2 ];
			c[ cix + 3*n2 ] = a[ aix+3 ];
			c[ cix + 4*n2 ] = a[ aix+4 ];
			c[ cix + 5*n2 ] = a[ aix+5 ];
			c[ cix + 6*n2 ] = a[ aix+6 ];
			c[ cix + 7*n2 ] = a[ aix+7 ];
		}
	}

	private static int[] countNnzPerColumn(MatrixBlock in, int maxCol) {
		return countNnzPerColumn(in, 0, in.getNumRows(), maxCol);
	}

	private static int[] countNnzPerColumn(MatrixBlock in, int rl, int ru, int maxCol) {
		//initial pass to determine capacity (this helps to prevent
		//sparse row reallocations and mem inefficiency w/ skew
		int[] cnt = null;
		if(in.clen <= maxCol) {
			SparseBlock a = in.sparseBlock;
			cnt = new int[in.clen];
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) )
					countAgg(cnt, a.indexes(i), a.pos(i), a.size(i));
			}
		}
		return cnt;
	}

	public static int[] countNnzPerColumn(MatrixBlock in) {
		return countNnzPerColumn(in, 0, in.getNumRows());
	}

	public static int[] countNnzPerColumn(MatrixBlock in, int rl, int ru) {
		if(in.isInSparseFormat())
			return countNnzPerColumnSparse(in, rl, ru);
		else
			return countNnzPerColumnDense(in, rl, ru);
	}

	private static int[] countNnzPerColumnSparse(MatrixBlock in, int rl, int ru) {
		final int[] cnt = new int[in.clen];
		final SparseBlock a = in.sparseBlock;
		for(int i = rl; i < ru; i++) {
			if(!a.isEmpty(i))
				countAgg(cnt, a.indexes(i), a.pos(i), a.size(i));
		}
		return cnt;
	}


	private static int[] countNnzPerColumnDense(MatrixBlock in, int rl, int ru) {
		final int[] cnt = new int[in.clen];
		final double[] dV = in.getDenseBlockValues();
		int off = rl * in.clen;
		for(int i = rl; i < ru; i++)
			for(int j = 0; j < in.clen; j++)
				if(dV[off++] != 0)
					cnt[j]++;
			
		return cnt;
	}

	public static int[] mergeNnzCounts(int[] cnt, int[] cnt2) {
		if( cnt == null )
			return cnt2;
		for( int i=0; i<cnt.length; i++ )
			cnt[i] += cnt2[i];
		return cnt;
	}

	private static void reverseDense(MatrixBlock in, MatrixBlock out) {
		final int m = in.rlen;
		final int n = in.clen;
		
		//set basic meta data and allocate output
		out.sparse = false;
		out.nonZeros = in.nonZeros;
		out.allocateDenseBlock(false);
		
		//copy all rows into target positions
		if( n == 1 ) { //column vector
			double[] a = in.getDenseBlockValues();
			double[] c = out.getDenseBlockValues();
			for( int i=0; i<m; i++ )
				c[m-1-i] = a[i];
		}
		else { //general matrix case
			DenseBlock a = in.getDenseBlock();
			DenseBlock c = out.getDenseBlock();
			for( int i=0; i<m; i++ ) {
				final int ri = m - 1 - i;
				System.arraycopy(a.values(i), a.pos(i), c.values(ri), c.pos(ri), n);
			}
		}
	}

	private static void reverseSparse(MatrixBlock in, MatrixBlock out) {
		final int m = in.rlen;
		
		//set basic meta data and allocate output
		out.sparse = true;
		out.nonZeros = in.nonZeros;
		
		out.allocateSparseRowsBlock(false);
		
		//copy all rows into target positions
		SparseBlock a = in.getSparseBlock();
		SparseBlock c = out.getSparseBlock();
		for( int i=0; i<m; i++ )
			if( !a.isEmpty(i) )
				c.set(m-1-i, a.get(i), true);
	}

	private static void rollDense(MatrixBlock in, MatrixBlock out, int shift) {
		final int m = in.rlen;
		shift %= (m != 0 ? m : 1); // roll matrix with axis=none

		copyDenseMtx(in, out, 0, shift, m - shift, false, true);
		copyDenseMtx(in, out, m - shift, 0, shift, true, true);
	}

	private static void rollSparse(MatrixBlock in, MatrixBlock out, int shift) {
		final int m = in.rlen;
		shift %= (m != 0 ? m : 1); // roll matrix with axis=0

		copySparseMtx(in, out, 0, shift, m - shift, false, true);
		copySparseMtx(in, out, m-shift, 0, shift, false, true);
	}

	public static void copyMtx(MatrixBlock in, MatrixBlock out, int inStart, int outStart, int copyLen,
							   boolean isAllocated, boolean copyTotalNonZeros) {
		if (in.isInSparseFormat()){
			copySparseMtx(in, out, inStart, outStart, copyLen, isAllocated, copyTotalNonZeros);
		} else {
			copyDenseMtx(in, out, inStart, outStart, copyLen, isAllocated, copyTotalNonZeros);
		}
	}

	public static void copyDenseMtx(MatrixBlock in, MatrixBlock out, int inIdx, int outIdx, int copyLen,
									boolean isAllocated, boolean copyTotalNonZeros) {
		int clen = in.clen;

		// set basic meta data and allocate output
		if (!isAllocated){
			out.sparse = false;
			if (copyTotalNonZeros) out.nonZeros = in.nonZeros;
			out.allocateDenseBlock(false);
		}

		// copy all rows into target positions
		if (clen == 1) { //column vector
			double[] a = in.getDenseBlockValues();
			double[] c = out.getDenseBlockValues();

			System.arraycopy(a, inIdx, c, outIdx, copyLen);
		} else {
			DenseBlock a = in.getDenseBlock();
			DenseBlock c = out.getDenseBlock();

			while (copyLen > 0) {
				System.arraycopy(a.values(inIdx), a.pos(inIdx),
						c.values(outIdx), c.pos(outIdx), clen);

				inIdx++; outIdx++; copyLen--;
			}
		}
	}

	private static void copySparseMtx(MatrixBlock in, MatrixBlock out, int inIdx, int outIdx, int copyLen,
									  boolean isAllocated, boolean copyTotalNonZeros) {
		//set basic meta data and allocate output
		if (!isAllocated){
			out.sparse = true;
			if (copyTotalNonZeros) out.nonZeros = in.nonZeros;
			out.allocateSparseRowsBlock(false);
		}

		SparseBlock a = in.getSparseBlock();
		SparseBlock c = out.getSparseBlock();

		for (int i = 0; i < copyLen; i++) {
			if (!a.isEmpty(inIdx)){
				final int apos = a.pos(inIdx);
				final int alen = a.size(inIdx) + apos;
				final int[] aix = a.indexes(inIdx);
				final double[] avals = a.values(inIdx);

				// copy only non-zero elements
				for (int k = apos; k < alen; k++)
					c.set(outIdx, aix[k], avals[k]);
			}
			inIdx++; outIdx++;
		}
	}

	/**
	 * Generic implementation diagV2M
	 * (in most-likely DENSE, out most likely SPARSE)
	 * 
	 * @param in input matrix
	 * @param out output matrix
	 */
	private static void diagV2M( MatrixBlock in, MatrixBlock out )
	{
		final int rlen = in.rlen;
		
		//CASE column vector
		if( out.sparse  ) { //SPARSE
			if( SPARSE_OUTPUTS_IN_CSR ) {
				int[] rptr = new int[in.rlen+1];
				int[] cix = null;
				double[] vals = null;
				//case a: fully dense vector
				if( rlen == in.nonZeros && !in.sparse ) {
					//reuse single seq for rptr and cix (cix truncated by 1)
					rptr = cix = UtilFunctions.getSeqArray(0, rlen, 1);
					vals = in.getDenseBlockValues(); //shallow copy
				}
				//case b: more general input
				else {
					cix = new int[(int)in.nonZeros];
					vals = new double[(int)in.nonZeros];
					for( int i=0, pos=0; i<rlen; i++ ) {
						double val = in.get(i, 0);
						if( val != 0 ) {
							cix[pos] = i;
							vals[pos] = val;
							pos++;
						}
						rptr[i+1]=pos;
					}
				}
				out.sparseBlock = new SparseBlockCSR(
					rptr, cix, vals, (int)in.nonZeros);
			}
			else {
				out.allocateBlock();
				SparseBlock sblock = out.sparseBlock;
				for(int i=0; i<rlen; i++) {
					double val = in.get(i, 0);
					if( val != 0 ) {
						sblock.allocate(i, 1);
						sblock.append(i, i, val);
					}
				}
			}
		}
		else { //DENSE
			for( int i=0; i<rlen; i++ ) {
				double val = in.get(i, 0);
				if( val != 0 )
					out.appendValue(i, i, val);
			}
		}
		
		out.setNonZeros(in.nonZeros);
	}
	
	/**
	 * Generic implementation diagM2V (non-performance critical)
	 * (in most-likely SPARSE, out most likely DENSE)
	 * 
	 * NOTE: squared block assumption (checked on entry diag)
	 * 
	 * @param in input matrix
	 * @param out output matrix
	 */
	private static void diagM2V( MatrixBlock in, MatrixBlock out )
	{
		DenseBlock c = out.allocateBlock().getDenseBlock();
		int rlen = in.rlen;
		int nnz = 0;
		for( int i=0; i<rlen; i++ ) {
			double val = in.get(i, i);
			if( val != 0 ) {
				c.set(i, 0, val);
				nnz++;
			}
		}
		out.setNonZeros(nnz);
	}

	private static void reshapeDense( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise ) {
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.denseBlock == null )
			return;

		//shallow dense by-row reshape (w/o result allocation)
		if( SHALLOW_COPY_REORG && rowwise && in.denseBlock.numBlocks()==1 ) {
			//since the physical representation of dense matrices is always the same,
			//we don't need to create a copy, given our copy on write semantics.
			//however, note that with update in-place this would be an invalid optimization
			out.denseBlock = DenseBlockFactory.createDenseBlock(in.getDenseBlockValues(), rows, cols);
			return;
		}
		
		//allocate block if necessary
		out.allocateDenseBlock(false);
		
		//dense reshape
		DenseBlock a = in.getDenseBlock();
		DenseBlock c = out.getDenseBlock();
		
		if( rowwise ) {
			//VECTOR-MATRIX, MATRIX-VECTOR, GENERAL CASE
			//pure copy of rowwise internal representation
			c.set(a);
		}
		else { //colwise
			if( rlen==1 || clen==1 ) { //VECTOR->MATRIX
				//note: cache-friendly on a but not on c
				double[] avals = a.valuesAt(0);
				double[] cvals = c.valuesAt(0);
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0, cix=0; i<rows; i++, cix+=cols )
						cvals[ cix + j ] = avals[ aix++ ];
			}
			else if( rows==1 || cols==1 ) //MATRIX->VECTOR
			{
				//note: cache-friendly on c but not on a
				double[] avals = a.valuesAt(0);
				double[] cvals = c.valuesAt(0);
				for( int j=0, cix=0; j<clen; j++ )
					for( int i=0, aix=0; i<rlen; i++, aix+=clen )
						cvals[ cix++ ] = avals[ aix + j ];
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on c but not an a
				for( int i=0; i<rows; i++ ) {
					double[] cvals = c.values(i);
					int cix = c.pos(i);
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows ) {
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						cvals[cix+j] = a.get(ai,aj);
					}
				}
				//index conversion c[i,j]<- a[k,l]:
				// k = (rows*j+i)%rlen
				// l = (rows*j+i)/rlen
			}
		}
	}

	private static void reshapeSparse(MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise, int k)
		throws Exception {
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.isEmptyBlock(false) )
			return;
		
		//allocate block if necessary
		out.allocateSparseRowsBlock(false);
		int estnnz = (int) (in.nonZeros/rows);
		
		//sparse reshape
		SparseBlock a = in.sparseBlock;
		SparseBlock c = out.sparseBlock;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix not really different from general
			// * clen=1 and cols=1 will never be sparse.
			
			if( rows==1 ) //MATRIX->VECTOR
			{
				//note: cache-friendly on a and c; append-only
				c.allocate(0, estnnz, cols);
				for( int i=0, cix=0; i<rlen; i++, cix+=clen ) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ )
						c.append(0, cix+aix[j], avals[j]);
				}
			}
			else if( cols%clen==0 // SPECIAL CSR N:1 MATRIX->MATRIX
				&& SHALLOW_COPY_REORG && SPARSE_OUTPUTS_IN_CSR && in.getNonZeros() < Integer.MAX_VALUE) { // int nnz
				reshapeSparseToCSR(in, out, rows, cols);
			}
			else
				reshapeSparseToMCSR(in, out, rows, cols, k);
		}	
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			// * clen=1 and cols=1 will never be sparse.
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				if( !a.isEmpty(0) ){
					int alen = a.size(0); //always pos 0
					int[] aix = a.indexes(0);
					double[] avals = a.values(0);
					for( int j=0; j<alen; j++ ) {
						int ci = aix[j]%rows;
						int cj = aix[j]/rows;
						c.allocate(ci, estnnz, cols);
						c.append(ci, cj, avals[j]);
					}
				}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c; append&sort, in-place w/o shifts
				for( int i=0; i<rlen; i++ ) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						//long tmpix because total cells in sparse can be larger than int
						long tmpix = (long)aix[j]*rlen+i;
						int ci = (int)(tmpix%rows);
						int cj = (int)(tmpix/rows); 
						c.allocate(ci, estnnz, cols);
						c.append(ci, cj, avals[j]);
					}
				}
				out.sortSparseRows();
			}
		}
	}

	private static void reshapeSparseToMCSR(MatrixBlock in, MatrixBlock out, int rows, int cols, int k) throws Exception{
		int rlen = in.rlen;
		int clen = in.clen;

		// allocate block
		out.allocateSparseRowsBlock(false);
		int estnnz = (int) (in.nonZeros / rows);

		// sparse reshape
		SparseBlock a = in.sparseBlock;
		SparseBlock c = out.sparseBlock;
		if(cols % clen == 0)
			reshapeSparseToMCSR_Nto1(in, out, rows, cols, k);
		else // GENERAL CASE: MATRIX->MATRIX
		{
			// note: cache-friendly on a but not c; append-only
			// long cix because total cells in sparse can be larger than int
			long cix = 0;
			for(int i = 0; i < rlen; i++) {
				if(!a.isEmpty(i)) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for(int j = apos; j < apos + alen; j++) {
						int ci = (int) ((cix + aix[j]) / cols);
						int cj = (int) ((cix + aix[j]) % cols);
						c.allocate(ci, estnnz, cols);
						c.append(ci, cj, avals[j]);
					}
				}

				cix += clen;
			}
		}
	}

	private static void reshapeSparseToMCSR_Nto1(MatrixBlock in, MatrixBlock out, int rows, int cols, int k) throws Exception{
		// SPECIAL N:1 MATRIX->MATRIX
		final int rlen = in.rlen;
		final int clen = in.clen;

		final SparseBlock a = in.sparseBlock;
		final SparseBlock c = out.sparseBlock;
		final int n = cols / clen;
		// safe now since we fixed the parfor threading.
		if((k > 1 || k == -1) && ((double) rows / k) > 16) {
			final ExecutorService pool = CommonThreadPool.get(k == -1 ? InfrastructureAnalyzer.getLocalParallelism() : k);
			try {
				final int blkz = Math.max((int) Math.ceil((double) rows / k), 16);
				ArrayList<Future<?>> tasks = new ArrayList<>();
				for(int i = 0; i < rows; i += blkz) {
					final int start = i;
					final int end = Math.min(i + blkz, rows);
					tasks.add(pool.submit(() -> {
						for(int bi = start * n, ci = start; ci < end; bi += n, ci++) {
							reshapeSparseToMCSR_Nto1_row(a, c, clen, bi, n, ci);
						}
					}));
				}
				for(Future<?> f : tasks)
					f.get();
			}
			finally {
				pool.shutdown();
			}
		}
		else {
			for(int bi = 0, ci = 0; bi < rlen; bi += n, ci++) {
				reshapeSparseToMCSR_Nto1_row(a, c, clen, bi, n, ci);
			}
		}

		out.setNonZeros(in.nonZeros);
	}

	private static void reshapeSparseToMCSR_Nto1_row(SparseBlock a, SparseBlock c, int clen, int bi, int n, int ci){
		// allocate output row once (w/o re-allocations)
		final int s = (int) a.size(bi, bi + n); // get exact size of row output
		final int[] cix = new int[s];
		final double[] cvals =  new double[s];
		
		int pos = 0;
		// copy N input rows into output row
		for(int i = bi, colOffset = 0; i < bi + n; i++, colOffset += clen) {
			pos = reshapeSparseToMCSR_Nto1_row_one(a,i, pos,cix, colOffset, cvals);
		}
		c.set(ci, new SparseRowVector(cvals, cix), false);
	}

	private static int reshapeSparseToMCSR_Nto1_row_one(SparseBlock a, int i, int pos, int[] cix, int colOffset,
		double[] cvals) {
		if(a.isEmpty(i))
			return pos;
		final int apos = a.pos(i);
		final int alen = a.size(i);
		final int[] aix = a.indexes(i);
		final double[] avals = a.values(i);
		for(int j = apos; j < apos + alen; j++, pos++) {
			cix[pos] = colOffset + aix[j];
			cvals[pos] = avals[j];
		}
		return pos;
	}

	private static void reshapeSparseToCSR(MatrixBlock in, MatrixBlock out, int rows, int cols) {
		if(in.sparseBlock instanceof SparseBlockCSR) 
			reshapeSparseToCSRFromCSR(in, out, rows, cols);
		else 
			reshapeSparseToCSRFromMCSR(in, out, rows, cols);
	}

	private static void reshapeSparseToCSRFromMCSR(MatrixBlock in, MatrixBlock out, int rows, int cols) {
		final SparseBlock a = in.sparseBlock;
		final int rlen = in.rlen;
		final int clen = in.clen;

		final int n = cols / clen;
		final int[] rptr = new int[rows + 1];
		final int[] indexes = new int[(int) a.size()];
		final double[] values = new double[indexes.length];
		
		int pos = 0;

		for(int bi = 0, ci = 0; bi < rlen; bi += n, ci++) { // output rows
			for(int i = bi, cix = 0; i < bi + n; i++, cix += clen) { // N input rows
				if(a.isEmpty(i))
					continue;
				final int apos = a.pos(i);
				final int alen = a.size(i);
				final int[] aix = a.indexes(i);
				System.arraycopy(a.values(i), apos, values, pos, alen);
				for(int j = apos; j < apos + alen; j++)
					indexes[pos++] = cix + aix[j];
			}
			rptr[ci + 1] = pos;
		}

		// create CSR block from constructed or shallow-copy arrays
		out.sparseBlock = new SparseBlockCSR(rptr, indexes, values, pos);
	}


	private static void reshapeSparseToCSRFromCSR(MatrixBlock in, MatrixBlock out, int rows, int cols) {		
		final SparseBlock a = in.sparseBlock;
		int rlen = in.rlen;
		int clen = in.clen;

		int n = cols / clen, pos = 0;
		int[] rptr = new int[rows + 1];
		int[] indexes = new int[(int) a.size()];
		double[] values = null;
		rptr[0] = 0;

		int[] aix = ((SparseBlockCSR) a).indexes();
		for(int bi = 0, ci = 0; bi < rlen; bi += n, ci++) {
			for(int i = bi, cix = 0; i < bi + n; i++, cix += clen) {
				if(a.isEmpty(i))
					continue;
				int apos = a.pos(i);
				int alen = a.size(i);
				for(int j = apos; j < apos + alen; j++)
					indexes[pos++] = cix + aix[j];
			}
			rptr[ci + 1] = pos;
		}
		// shallow copy of CSR values
		values = ((SparseBlockCSR) a).values();

		// create CSR block from constructed or shallow-copy arrays
		out.sparseBlock = new SparseBlockCSR(rptr, indexes, values, pos);
	}

	private static void reshapeDenseToSparse( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise )
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.denseBlock == null )
			return;
		
		//allocate block if necessary
		out.allocateSparseRowsBlock(false);
		int estnnz = (int) (in.nonZeros/rows);
		
		//sparse reshape
		DenseBlock a = in.getDenseBlock();
		SparseBlock c = out.sparseBlock;
		
		if( rowwise ) {
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c; append-only
			for( int i=0; i<rlen; i++ ) {
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<clen; j++ ) {
					double val = avals[aix+j];
					if( val != 0 ) {
						long cix = (long) i*clen+j;
						int ci = (int)(cix / cols);
						int cj = (int)(cix % cols);
						c.allocate(ci, estnnz, cols);
						c.append(ci, cj, val);
					}
				}
			}
		}
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				double[] avals = a.valuesAt(0);
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0; i<rows; i++ ) {
						double val = avals[aix++];
						if( val != 0 ) {
							c.allocate(i, estnnz, cols);
							c.append(i, j, val);
						}
					}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on c but not a; append-only
				for( int i=0; i<rows; i++ )
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows ) {
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						double val = a.get(ai, aj);
						if( val != 0 ) {
							c.allocate(i, estnnz, cols);
							c.append(i, j, val);
						}
					}
			}
		}
	}

	private static void reshapeSparseToDense( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise ) {
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.sparseBlock == null )
			return;
		
		//allocate block if necessary
		out.allocateDenseBlock(false);
		
		//sparse/dense reshape
		SparseBlock a = in.sparseBlock;
		DenseBlock c = out.getDenseBlock();
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c
			for( int i=0, cix=0; i<rlen; i++, cix+=clen ) {
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						int ci = (cix+aix[j]) / cols;
						int cj = (cix+aix[j]) % cols;
						c.set(ci, cj, avals[j]);
					}
				}
			}
		}
		else //colwise
		{
			//NOTES on special cases
			// * matrix-vector not really different from general
			
			if( rlen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not c
				double[] cvals = c.valuesAt(0);
				if( !a.isEmpty(0) ) {
					int apos = a.pos(0);
					int alen = a.size(0);
					int[] aix = a.indexes(0);
					double[] avals = a.values(0);
					for( int j=apos; j<apos+alen; j++ ) {
						int ci = aix[j]%rows;
						int cj = aix[j]/rows;
						cvals[ci*cols+cj] = avals[j];
					}
				}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c
				for( int i=0; i<rlen; i++ ) {
					if( a.isEmpty(i) ) continue;
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						int tmpix = aix[j]*rlen+i;
						int ci = tmpix%rows;
						int cj = tmpix/rows;
						c.set(ci, cj, avals[j]);
					}
				}
			}
		}
	}
	
	///////////////////////////////
	// private MR implementation //
	///////////////////////////////

	private static Collection<MatrixIndexes> computeAllResultBlockIndexes(MatrixIndexes ixin,
		DataCharacteristics mcIn, DataCharacteristics mcOut, MatrixBlock in, boolean rowwise, boolean outputEmpty )
	{
		HashSet<MatrixIndexes> ret = new HashSet<>();
		
		long row_offset = (ixin.getRowIndex()-1)*mcOut.getBlocksize();
		long col_offset = (ixin.getColumnIndex()-1)*mcOut.getBlocksize();
		long max_row_offset = Math.min(mcIn.getRows(),row_offset+mcIn.getBlocksize())-1;
		long max_col_offset = Math.min(mcIn.getCols(),col_offset+mcIn.getBlocksize())-1;
		
		if( rowwise ) {
			if( mcIn.getCols() == 1 ) {
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), row_offset, 0, mcIn, mcOut, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), max_row_offset, 0, mcIn, mcOut, rowwise);
				createRowwiseIndexes(first, last, mcOut.getNumColBlocks(), ret);
			}
			else if( in.getNonZeros()<in.getNumRows() && !outputEmpty ) {
				createNonZeroIndexes(mcIn, mcOut, in, row_offset, col_offset, rowwise, ret);
			}
			else {
				for( long i=row_offset; i<max_row_offset+1; i++ ) {
					MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), i, col_offset, mcIn, mcOut, rowwise);
					MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), i, max_col_offset, mcIn, mcOut, rowwise);
					createRowwiseIndexes(first, last, mcOut.getNumColBlocks(), ret);
				}
			}
		}
		else{ //colwise
			if( mcIn.getRows() == 1 ) {
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), 0, col_offset, mcIn, mcOut, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), 0, max_col_offset, mcIn, mcOut, rowwise);
				createColwiseIndexes(first, last, mcOut.getNumRowBlocks(), ret);
			}
			else if( in.getNonZeros()<in.getNumColumns() && !outputEmpty ) {
				createNonZeroIndexes(mcIn, mcOut, in, row_offset, col_offset, rowwise, ret);
			}
			else {
				for( long j=col_offset; j<max_col_offset+1; j++ ) {
					MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), row_offset, j, mcIn, mcOut, rowwise);
					MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), max_row_offset, j, mcIn, mcOut, rowwise);
					createColwiseIndexes(first, last, mcOut.getNumRowBlocks(), ret);
				}
			}
		}
		return ret;
	}
	
	private static void createRowwiseIndexes(MatrixIndexes first, MatrixIndexes last, long ncblks, HashSet<MatrixIndexes> ret) {
		if( first.getRowIndex()<=0 || first.getColumnIndex()<=0 )
			throw new RuntimeException("Invalid computed first index: "+first.toString());
		if( last.getRowIndex()<=0 || last.getColumnIndex()<=0 )
			throw new RuntimeException("Invalid computed last index: "+last.toString());
		
		//add first row block
		ret.add(first);
		
		//add blocks in between first and last
		if( !first.equals(last) ) {
			boolean fill = first.getRowIndex()==last.getRowIndex()
				&& first.getColumnIndex() > last.getColumnIndex();
			for( long k1=first.getRowIndex(); k1<=last.getRowIndex(); k1++ ) {
				long k2_start = (k1==first.getRowIndex() ? first.getColumnIndex()+1 : 1);
				long k2_end = (k1==last.getRowIndex() && !fill) ? last.getColumnIndex()-1 : ncblks;
				for( long k2=k2_start; k2<=k2_end; k2++ )
					ret.add(new MatrixIndexes(k1,k2));
			}
			ret.add(last);
		}
	}
	
	private static void createColwiseIndexes(MatrixIndexes first, MatrixIndexes last, long nrblks, HashSet<MatrixIndexes> ret) {
		if( first.getRowIndex()<=0 || first.getColumnIndex()<=0 )
			throw new RuntimeException("Invalid computed first index: "+first.toString());
		if( last.getRowIndex()<=0 || last.getColumnIndex()<=0 )
			throw new RuntimeException("Invalid computed last index: "+last.toString());
		
		//add first row block
		ret.add(first);
		
		//add blocks in between first and last
		if( !first.equals(last) ) {
			boolean fill = first.getColumnIndex()==last.getColumnIndex()
					&& first.getRowIndex() > last.getRowIndex();
			for( long k1=first.getColumnIndex(); k1<=last.getColumnIndex(); k1++ ) {
				long k2_start = ((k1==first.getColumnIndex()) ? first.getRowIndex()+1 : 1);
				long k2_end = ((k1==last.getColumnIndex() && !fill) ? last.getRowIndex()-1 : nrblks);
				for( long k2=k2_start; k2<=k2_end; k2++ )
					ret.add(new MatrixIndexes(k2,k1));
			}
			ret.add(last);
		}
	}
	
	private static void createNonZeroIndexes(DataCharacteristics mcIn, DataCharacteristics mcOut,
	                                         MatrixBlock in, long row_offset, long col_offset, boolean rowwise, HashSet<MatrixIndexes> ret) {
		Iterator<IJV> iter = in.getSparseBlockIterator();
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			ret.add(computeResultBlockIndex(new MatrixIndexes(),
				row_offset+cell.getI(), col_offset+cell.getJ(), mcIn, mcOut, rowwise));
		}
	}
	
	private static Map<MatrixIndexes, MatrixBlock> createAllResultBlocks(Collection<MatrixIndexes> rix, long nnz, DataCharacteristics mcOut) {
		return rix.stream().collect(Collectors.toMap(ix -> ix, ix -> createResultBlock(ix, nnz, rix.size(), mcOut)));
	}
	
	private static MatrixBlock createResultBlock(MatrixIndexes ix, long nnz, int nBlocks, DataCharacteristics mcOut) {
		//compute indexes
		long bi = ix.getRowIndex();
		long bj = ix.getColumnIndex();
		int lbrlen = UtilFunctions.computeBlockSize(mcOut.getRows(), bi, mcOut.getBlocksize());
		int lbclen = UtilFunctions.computeBlockSize(mcOut.getCols(), bj, mcOut.getBlocksize());
		if( lbrlen<1 || lbclen<1 )
			throw new DMLRuntimeException("Computed block dimensions ("+bi+","+bj+" -> "+lbrlen+","+lbclen+") are invalid!");
		//create result block
		int estnnz = (int) (nnz/nBlocks); //force initial capacity per row to 1, for many blocks
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(lbrlen, lbclen, estnnz);
		return new MatrixBlock(lbrlen, lbclen, sparse, estnnz); 
	}

	private static void reshapeDense(MatrixBlock in, long row_offset, long col_offset, Map<MatrixIndexes,MatrixBlock> rix,
	                                 DataCharacteristics mcIn, DataCharacteristics mcOut, boolean rowwise ) {
		if( in.isEmptyBlock(false) )
			return;
		
		int rlen = in.rlen;
		int clen = in.clen;
		double[] a = in.getDenseBlockValues();
		
		//append all values to right blocks
		MatrixIndexes ixtmp = new MatrixIndexes();
		for( int i=0, aix=0; i<rlen; i++, aix+=clen )
		{
			long ai = row_offset+i;
			for( int j=0; j<clen; j++ )
			{
				double val = a[ aix+j ];
				if( val !=0 ) {
					long aj = col_offset+j;
					ixtmp = computeResultBlockIndex(ixtmp, ai, aj, mcIn, mcOut, rowwise);
					MatrixBlock out = rix.get(ixtmp);
					if( out == null )
						throw new DMLRuntimeException("Missing result block: "+ixtmp);
					ixtmp = computeInBlockIndex(ixtmp, ai, aj, mcIn, mcOut, rowwise);
					out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), val);
				}
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise && mcIn.getRows() > 1 ) {
			rix.values().stream().filter(b -> b.sparse)
				.forEach(b -> b.sortSparseRows());
		}
	}

	private static void reshapeSparse(MatrixBlock in, long row_offset, long col_offset, Map<MatrixIndexes,MatrixBlock> rix,
	                                  DataCharacteristics mcIn, DataCharacteristics mcOut, boolean rowwise ) {
		if( in.isEmptyBlock(false) )
			return;
		
		int rlen = in.rlen;
		SparseBlock a = in.sparseBlock;
		
		//append all values to right blocks
		MatrixIndexes ixtmp = new MatrixIndexes();
		for( int i=0; i<rlen; i++ ) {
			if( a.isEmpty(i) ) continue;
			long ai = row_offset+i;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			for( int j=apos; j<apos+alen; j++ )  {
				long aj = col_offset+aix[j];
				ixtmp = computeResultBlockIndex(ixtmp, ai, aj, mcIn, mcOut, rowwise);
				MatrixBlock out = getAllocatedBlock(rix, ixtmp);
				ixtmp = computeInBlockIndex(ixtmp, ai, aj, mcIn, mcOut, rowwise);
				out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), avals[j]);
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise && mcIn.getRows() > 1 ) {
			rix.values().stream().filter(b -> b.sparse)
				.forEach(b -> b.sortSparseRows());
		}
	}
	
	private static MatrixBlock getAllocatedBlock(Map<MatrixIndexes,MatrixBlock> rix, MatrixIndexes ix) {
		MatrixBlock out = rix.get(ix);
		if( out == null )
			throw new DMLRuntimeException("Missing result block: "+ix);
		return out;
	}
	
	/**
	 * Assumes internal (0-begin) indices ai, aj as input; computes external block indexes (1-begin) 
	 * 
	 * @param ixout matrix indexes for reuse
	 * @param ai row index
	 * @param aj column index
	 * @param mcIn input matrix characteristics
	 * @param mcOut output matrix characteristics
	 * @param rowwise row-wise extraction
	 * @return matrix indexes
	 */
	private static MatrixIndexes computeResultBlockIndex(MatrixIndexes ixout, long ai, long aj,
		DataCharacteristics mcIn, DataCharacteristics mcOut, boolean rowwise )
	{
		long tempc = computeGlobalCellIndex(mcIn, ai, aj, rowwise);
		long ci = rowwise ? tempc/mcOut.getCols() : tempc%mcOut.getRows();
		long cj = rowwise ? tempc%mcOut.getCols() : tempc/mcOut.getRows();
		long bci = ci/mcOut.getBlocksize() + 1;
		long bcj = cj/mcOut.getBlocksize() + 1;
		return ixout.setIndexes(bci, bcj);
	}
	
	private static MatrixIndexes computeInBlockIndex(MatrixIndexes ixout, long ai, long aj,
	                                                 DataCharacteristics mcIn, DataCharacteristics mcOut, boolean rowwise )
	{
		long tempc = computeGlobalCellIndex(mcIn, ai, aj, rowwise);
		long ci = rowwise ? (tempc/mcOut.getCols())%mcOut.getBlocksize() : 
			(tempc%mcOut.getRows())%mcOut.getBlocksize();
		long cj = rowwise ? (tempc%mcOut.getCols())%mcOut.getBlocksize() : 
			(tempc/mcOut.getRows())%mcOut.getBlocksize();
		return ixout.setIndexes(ci, cj);
	}
	
	private static long computeGlobalCellIndex(DataCharacteristics mcIn, long ai, long aj, boolean rowwise) {
		return rowwise ? ai*mcIn.getCols()+aj : ai+mcIn.getRows()*aj;
	}

	private static MatrixBlock removeEmptyRows(MatrixBlock in, MatrixBlock ret, MatrixBlock select, boolean emptyReturn) {
		final int m = in.rlen;
		final int n = in.clen;
		boolean[] flags = null;
		int rlen2 = 0; 
		
		//Step 0: special case handling
		if( SHALLOW_COPY_REORG && SPARSE_OUTPUTS_IN_CSR
			&& in.sparse && !in.isEmptyBlock(false)
			&& select==null && in.sparseBlock instanceof SparseBlockCSR
			&& in.nonZeros < Integer.MAX_VALUE )
		{
			//create the output in csr format with a shallow copy of arrays for column 
			//indexes and values (heuristic: shallow copy better than copy to dense)
			SparseBlockCSR sblock = (SparseBlockCSR) in.sparseBlock;
			int lrlen = 0;
			for( int i=0; i<m; i++ )
				lrlen += sblock.isEmpty(i) ? 0 : 1;
			//check for sparse output representation, otherwise
			//fall back to default pass to allocate in dense format
			if( MatrixBlock.evalSparseFormatInMemory(lrlen, n, in.nonZeros) ) {
				int[] rptr = new int[lrlen+1];
				for( int i=0, j=0, pos=0; i<m; i++ )
					if( !sblock.isEmpty(i) ) {
						pos += sblock.size(i);
						rptr[++j] = pos;
					}
				ret.reset(lrlen, in.clen, true);
				ret.sparseBlock = new SparseBlockCSR(rptr,
					sblock.indexes(), sblock.values(), (int)in.nonZeros);
				ret.nonZeros = in.nonZeros;
				return ret;
			}
		}
		
		//Step 1: scan block and determine non-empty rows
		if(select == null) 
		{
			flags = new boolean[ m ]; //false
			if( in.sparse ) { //SPARSE 
				SparseBlock a = in.sparseBlock;
				for ( int i=0; i < m; i++ )
					rlen2 += (flags[i] = !a.isEmpty(i)) ? 1 : 0;
			}
			else { //DENSE
				DenseBlock a = in.getDenseBlock();
				for( int i=0; i<m; i++ ) {
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for(int j=0; j<n; j++)
						if( avals[aix+j] != 0 ) {
							flags[i] = true;
							rlen2++;
							//early abort for current row
							break; 
						}
				}
			}
		} 
		else {
			flags = DataConverter.convertToBooleanVector(select);
			rlen2 = (int)select.getNonZeros();
		}

		//Step 2: reset result and copy rows
		//dense stays dense if correct input representation (but robust for any input), 
		//sparse might be dense/sparse
		rlen2 = Math.max(rlen2, emptyReturn ? 1 : 0); //ensure valid output
		boolean sp = MatrixBlock.evalSparseFormatInMemory(rlen2, n, in.nonZeros);
		ret.reset(rlen2, n, sp);
		if( in.isEmptyBlock(false) )
			return ret;
		
		if( SHALLOW_COPY_REORG && m == rlen2 && select == null ) {
			// the condition m==rlen2 is not enough with non-empty 1-row input but empty 
			// 1-row select vector because if emptyReturn should output a single empty row
			ret.sparse = in.sparse;
			if( ret.sparse )
				ret.sparseBlock = in.sparseBlock;
			else
				ret.denseBlock = in.denseBlock;
		}
		else if( in.sparse ) //* <- SPARSE
		{
			//note: output dense or sparse
			for( int i=0, cix=0; i<m; i++ )
				if( flags[i] )
					ret.appendRow(cix++, in.sparseBlock.get(i), !SHALLOW_COPY_REORG);
		}
		else if( !in.sparse && !ret.sparse )  //DENSE <- DENSE
		{
			ret.allocateDenseBlock();
			DenseBlock a = in.getDenseBlock();
			DenseBlock c = ret.getDenseBlock();
			for( int i=0, ci=0; i<m; i++ )
				if( flags[i] ) {
					System.arraycopy(a.values(i),
						a.pos(i), c.values(ci), c.pos(ci), n);
					ci++; //target row index
				}
		}
		else //SPARSE <- DENSE
		{
			ret.allocateSparseRowsBlock();
			DenseBlock a = in.getDenseBlock();
			for( int i=0, ci=0; i<m; i++ )
				if( flags[i] ) {
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for( int j=0; j<n; j++ )
						ret.appendValue(ci, j, avals[aix+j]);
					ci++;
				}
		}
		
		//check sparsity
		ret.nonZeros = (select==null) ?
			in.nonZeros : ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	private static MatrixBlock removeEmptyColumns(MatrixBlock in, MatrixBlock ret, MatrixBlock select, boolean emptyReturn) {
		final int m = in.rlen;
		final int n = in.clen;
		
		//Step 1: scan block and determine non-empty columns 
		//(we optimized for cache-friendly behavior and hence don't do early abort)
		boolean[] flags = null; 
		
		if (select == null) 
		{
			flags = new boolean[ n ]; //false
			if( in.sparse ) { //SPARSE 
				SparseBlock a = in.sparseBlock;
				for( int i=0; i<m; i++ ) 
					if ( !a.isEmpty(i) ) {
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						for( int j=apos; j<apos+alen; j++ )
							flags[ aix[j] ] = true;
					}
			}
			else { //DENSE
				DenseBlock a = in.getDenseBlock();
				for( int i=0; i<m; i++ ) {
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for( int j=0; j<n; j++ )
						flags[j] |= (avals[aix+j] != 0);
				}
			}
		} 
		else {
			flags = DataConverter.convertToBooleanVector(select);
		}
		
		//Step 2: determine number of columns
		int clen2 = 0;
		for( int j=0; j<n; j++ ) {
			clen2 += flags[j] ? 1 : 0;
		}
		
		//Step 3: reset result and copy columns
		//dense stays dense if correct input representation (but robust for any input), 
		// sparse might be dense/sparse
		clen2 = Math.max(clen2, emptyReturn ? 1 : 0); //ensure valid output
		boolean sp = MatrixBlock.evalSparseFormatInMemory(m, clen2, in.nonZeros);
		ret.reset(m, clen2, sp);
		if( in.isEmptyBlock(false) )
			return ret;
		
		if( SHALLOW_COPY_REORG && n == clen2 ) {
			//quick path: shallow copy if unmodified
			ret.sparse = in.sparse;
			if( ret.sparse )
				ret.sparseBlock = in.sparseBlock;
			else
				ret.denseBlock = in.denseBlock;
		}
		else
		{
			//create mapping of flags to target indexes
			int[] cix = new int[n];
			for( int j=0, pos=0; j<n; j++ ) {
				if( flags[j] )
					cix[j] = pos++;
			}
			
			//deep copy of modified outputs
			if( in.sparse ) //* <- SPARSE
			{
				//note: output dense or sparse
				SparseBlock a = in.sparseBlock;
				for( int i=0; i<m; i++ )
					if ( !a.isEmpty(i) ) {
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						for( int j=apos; j<apos+alen; j++ )
							if( flags[aix[j]] )
								ret.appendValue(i, cix[aix[j]], avals[j]);
					}
			}
			else if( !in.sparse && !ret.sparse ) { //DENSE <- DENSE
				ret.allocateDenseBlock();
				DenseBlock a = in.getDenseBlock();
				DenseBlock c = ret.getDenseBlock();
				for( int i=0; i<m; i++ ) {
					double[] avals = a.values(i);
					double[] cvals = c.values(i);
					int aix = a.pos(i);
					int lcix = c.pos(i);
					for( int j=0; j<n; j++ )
						if( flags[j] )
							 cvals[ lcix+cix[j] ] = avals[aix+j];
				}
			}
			else { //SPARSE <- DENSE
				ret.allocateSparseRowsBlock();
				DenseBlock a = in.getDenseBlock();
				for( int i=0; i<m; i++ ) {
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for( int j=0; j<n; j++ ) {
						double aval = avals[aix+j];
						if( flags[j] && aval!=0 )
							 ret.appendValue(i, cix[j], aval);
					}
				}
			}
		}
		
		//check sparsity
		ret.nonZeros = (select==null) ?
			in.nonZeros : ret.recomputeNonZeros();
		ret.examSparsity();
		
		return ret;
	}

	private static MatrixBlock rexpandRows(MatrixBlock in, MatrixBlock ret, int max, boolean cast, boolean ignore) {
		//set meta data
		final int rlen = max;
		final int clen = in.rlen;
		final long nnz = in.nonZeros;
		boolean sp = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz);
		ret.reset(rlen, clen, sp);

		//setup temporary array for 'buffered append w/ sorting' in order
		//to mitigate performance issues due to random row access for large m
		final int blksize = 1024*1024; //max 12MB
		int[] tmpi = new int[Math.min(blksize,clen)];
		double[] tmp = new double[Math.min(blksize,clen)];
		
		//expand input vertically  (input vector likely dense 
		//but generic implementation for general case)
		for( int i=0; i<clen; i+=blksize )
		{
			//create sorted block indexes (append buffer)
			int len = Math.min(blksize, clen-i);
			copyColVector(in, i, tmp, tmpi, len);
			SortUtils.sortByValue(0, len, tmp, tmpi);
		
			//process current append buffer
			for( int j=0; j<len; j++ )
			{
				//get value and cast if necessary (table)
				double val = tmp[j];
				if( cast )
					val = UtilFunctions.toLong(val);
				
				//handle invalid values if not to be ignored
				if( !ignore && val<=0 )
					throw new DMLRuntimeException("Invalid input value <= 0 for ignore=false: "+val);
				
				//set expanded value if matching
				//note: tmpi populated with i+j indexes, then sorted
				if( val == Math.floor(val) && val >= 1 && val <= max )
					ret.appendValue((int)(val-1), tmpi[j], 1);
			}
		}
		
		//ensure valid output sparse representation 
		//(necessary due to cache-conscious processing w/ unstable sort)
		if( ret.isInSparseFormat() )
			ret.sortSparseRows();
		
		return ret;
	}
	
	private static MatrixBlock rexpandColumns(MatrixBlock in, MatrixBlock ret, int max, boolean cast, boolean ignore, int k) {
		//set meta data
		final int rlen = in.rlen;
		final int clen = max;
		final long nnz = in.nonZeros;
		boolean sp = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz);
		ret.reset(rlen, clen, sp);
		ret.allocateBlock();
		
		//execute rexpand columns
		long rnnz = 0; //real nnz (due to cutoff max)
		if( k <= 1 || in.getNumRows() <= PAR_NUMCELL_THRESHOLD
			|| (sp && SPARSE_OUTPUTS_IN_CSR) ) {
			rnnz = rexpandColumns(in, ret, max, cast, ignore, 0, rlen);
		}
		else {
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<RExpandColsTask> tasks = new ArrayList<>();
				int blklen = (int)(Math.ceil((double)rlen/k/8));
				for( int i=0; i<8*k & i*blklen<rlen; i++ )
					tasks.add(new RExpandColsTask(in, ret, 
						max, cast, ignore, i*blklen, Math.min((i+1)*blklen, rlen)));

				for( Future<Long> task : pool.invokeAll(tasks) )
					rnnz += task.get();
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		}
		
		//post-processing
		ret.setNonZeros(rnnz);
		
		return ret;
	}

	private static long rexpandColumns(MatrixBlock in, MatrixBlock ret, int max, boolean cast, boolean ignore, int rl, int ru) {
		//initialize auxiliary data structures 
		int lnnz = 0;
		int[] cix = null;
		if( ret.sparse && SPARSE_OUTPUTS_IN_CSR ) {
			cix = new int[in.rlen];
			Arrays.fill(cix, -1);
		}
		
		//expand input horizontally (input vector likely dense 
		//but generic implementation for general case)
		DenseBlock cd = ret.getDenseBlock();
		SparseBlock cs = ret.getSparseBlock();
		for( int i=rl; i<ru; i++ )
		{
			//get value and cast if necessary (table)
			double val = in.get(i, 0);
			if( cast )
				val = UtilFunctions.toLong(val);
			
			//handle invalid values if not to be ignored
			if( !ignore && val<=0 )
				throw new DMLRuntimeException("Invalid input value <= 0 for ignore=false: "+val);
			
			//set expanded value if matching
			if( val == Math.floor(val) && val >= 1 && val <= max ) {
				//update target without global nnz maintenance
				if( cix != null ) {
					cix[i] = (int)(val-1);
				}
				else if( ret.sparse ) {
					cs.allocate(i, 1);
					cs.append(i, (int)(val-1), 1);
				}
				else
					cd.set(i, (int)(val-1), 1);
				lnnz ++;
			}
		}
		
		//init CSR block once to avoid repeated updates of row pointers on append
		if( cix != null )
			ret.sparseBlock = new SparseBlockCSR(in.rlen, lnnz, cix);
		
		//recompute nnz of partition
		return ret.setNonZeros(lnnz);
	}
	
	private static void copyColVector( MatrixBlock in, int ixin, double[] tmp, int[] tmpi, int len)
	{
		//copy value array from input matrix
		if( in.isEmptyBlock(false) ) {
			Arrays.fill(tmp, 0, len, 0);
		}
		else if( in.sparse ){ //SPARSE
			for( int i=0; i<len; i++ )
				tmp[i] = in.get(ixin+i, 0);
		}
		else { //DENSE
			System.arraycopy(in.getDenseBlockValues(), ixin, tmp, 0, len);
		}
		
		//init index array
		for( int i=0; i<len; i++ )
			tmpi[i] = ixin + i;
	}
	

	/**
	 * Utility method for in-place transformation of an ascending sorted
	 * order into a descending sorted order. This method assumes dense
	 * column vectors as input.
	 * 
	 * @param m1 matrix
	 */
	private static void sortReverseDense( MatrixBlock m1 ) {
		int rlen = m1.rlen;
		double[] a = m1.getDenseBlockValues();
		for( int i=0; i<rlen/2; i++ ) {
			double tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}

	private static void sortReverseDense( int[] a ) {
		int rlen = a.length;
		for( int i=0; i<rlen/2; i++ ) {
			int tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}

	private static void sortReverseDense( double[] a ) {
		int rlen = a.length;
		for( int i=0; i<rlen/2; i++ ) {
			double tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}

	// Method to merge all blocks of a specified length.
	private static void mergeSortedBlocks(int blockLength, int[] valueIndexes, double[] values, int k){
		// Check if the blocklength is bigger than the size of the values
		// if it is smaller then merge the blocks, if not merging is done

		int vlen = values.length;
		int mergeBlockSize = blockLength * 2;
		if (mergeBlockSize <= vlen + blockLength){
			ExecutorService pool = CommonThreadPool.get(k);
			try {
				ArrayList<MergeTask> tasks = new ArrayList<>();
				for( int i=0; i*mergeBlockSize<vlen; i++ ){
					int start = i*mergeBlockSize;
					if (start + blockLength < vlen){
						int stop = Math.min(vlen , (i+1)*mergeBlockSize);
						tasks.add(new MergeTask(start, stop, blockLength, valueIndexes, values));
					}
				}
				CommonThreadPool.invokeAndShutdown(pool, tasks);
				// recursive merge larger blocks.
				mergeSortedBlocks(mergeBlockSize, valueIndexes, values, k);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
			finally{
				pool.shutdown();
			}
		} 
	}
	
	private static void sortBySecondary(int rl, int ru, double[] values, int[] vix, MatrixBlock in, int[] by, int off) {
		//find runs of equal values in current offset and index range
		//replace value by next column, sort, and recurse until single value
		for( int i=rl; i<ru-1; i++ ) {
			double tmp = values[i];
			//determine run of equal values
			int len = 0;
			while( i+len+1<ru && tmp==values[i+len+1] )
				len++;
			//temp value replacement and recursive sort
			if( len > 0 ) {
				double old = values[i];
				//extract values of next column
				for(int j=i; j<i+len+1; j++)
					values[j] = in.get(vix[j], by[off]-1);
				//sort values, incl recursive decent
				SortUtils.sortByValue(i, i+len+1, values, vix);
				if( off+1 < by.length )
					sortBySecondary(i, i+len+1, values, vix, in, by, off+1);
				//reset values of previous level
				Arrays.fill(values, i, i+len+1, old);
				i += len; //skip processed run
			}
		}
	}
	
	private static void sortIndexesStable(int rl, int ru, double[] values, int[] vix, MatrixBlock in, int[] by, int off) {
		for( int i=rl; i<ru-1; i++ ) {
			double tmp = values[i];
			//determine run of equal values
			int len = 0;
			while( i+len+1<ru && tmp==values[i+len+1] )
				len++;
			//temp value replacement and recursive decent
			if( len > 0 ) {
				if( off < by.length ) {
					//extract values of next column
					for(int j=i; j<i+len+1; j++)
						values[j] = in.get(vix[j], by[off]-1);
					sortIndexesStable(i, i+len+1, values, vix, in, by, off+1);
				}
				else //unstable sort of run indexes (equal value guaranteed)
					Arrays.sort(vix, i, i+len+1);
				i += len; //skip processed run
			}
		}
	}
	
	private static boolean isValidSortByList(int[] by, int clen) {
		if( by == null || by.length==0 || by.length>clen )
			return false;
		for(int i=0; i<by.length; i++)
			if( by[i] <= 0 || clen < by[i])
				return false;
		return true;
	}

	@SuppressWarnings("unused")
	private static void countAgg( int[] c, int[] ai, final int len ) 
	{
		final int bn = len%8;
		
		//compute rest, not aligned to 8-block
		for( int i=0; i<bn; i++ )
			c[ ai[i] ]++;
		
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=bn; i<len; i+=8 )
		{
			c[ ai[ i+0 ] ] ++;
			c[ ai[ i+1 ] ] ++;
			c[ ai[ i+2 ] ] ++;
			c[ ai[ i+3 ] ] ++;
			c[ ai[ i+4 ] ] ++;
			c[ ai[ i+5 ] ] ++;
			c[ ai[ i+6 ] ] ++;
			c[ ai[ i+7 ] ] ++;
		}
	}
	
	private static void countAgg( int[] c, int[] aix, int ai, final int len ) 
	{
		final int bn = len%8;
		
		//compute rest, not aligned to 8-block
		for( int i=ai; i<ai+bn; i++ )
			c[ aix[i] ]++;
		
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=ai+bn; i<ai+len; i+=8 )
		{
			c[ aix[ i+0 ] ] ++;
			c[ aix[ i+1 ] ] ++;
			c[ aix[ i+2 ] ] ++;
			c[ aix[ i+3 ] ] ++;
			c[ aix[ i+4 ] ] ++;
			c[ aix[ i+5 ] ] ++;
			c[ aix[ i+6 ] ] ++;
			c[ aix[ i+7 ] ] ++;
		}
	}

	@SuppressWarnings("unused")
	private static class AscRowComparator implements Comparator<Integer> 
	{
		private MatrixBlock _mb = null;
		private int _col = -1;
		
		public AscRowComparator( MatrixBlock mb, int col )
		{
			_mb = mb;
			_col = col;
		}

		@Override
		public int compare(Integer arg0, Integer arg1) {
			double val0 = _mb.get(arg0, _col);
			double val1 = _mb.get(arg1, _col);
			return (val0 < val1 ? -1 : (val0 == val1 ? 0 : 1));
		}
	}


	private static MatrixBlock transposeSparseToSparseBlock(MatrixBlock in, int rl, int ru){
		final int nRow = in.getNumRows();
		final int nCol = in.getNumColumns();
		final SparseBlock a = in.getSparseBlock();
		final MatrixBlock ret = new MatrixBlock(nCol, ru - rl, true);
		final SparseBlockMCSR c = new SparseBlockMCSR(nCol, ru - rl);
		final SparseRow[] cs = c.getRows();
		final double sp = ((double) in.nonZeros) / nRow / nCol;
		final int est = (int)(sp * (ru - rl));
		for(int i = 0; i < nCol; i++)
			c.allocate(i, Math.max(2, est), ru - rl);
		
		for(int r = rl; r < ru; r++){
			if(a.isEmpty(r))
				continue;

			final int apos = a.pos(r);
			final int alen = a.size(r);
			final int[] aix = a.indexes(r);
			final double[] aval = a.values(r);
			final int off = r - rl;
			for(int j = apos; j < apos + alen; j++)
				cs[aix[j]] = cs[aix[j]].append(off, aval[j]);
			
		}
		ret.setSparseBlock(c);
		ret.recomputeNonZeros();
		return ret;
	}

	@SuppressWarnings("unused")
	private static class DescRowComparator implements Comparator<Integer> 
	{
		private MatrixBlock _mb = null;
		private int _col = -1;
		
		public DescRowComparator( MatrixBlock mb, int col )
		{
			_mb = mb;
			_col = col;
		}

		@Override
		public int compare(Integer arg0, Integer arg1) 
		{			
			double val0 = _mb.get(arg0, _col);
			double val1 = _mb.get(arg1, _col);
			return (val0 > val1 ? -1 : (val0 == val1 ? 0 : 1));
		}		
	}

	private static class TransposeTask implements Callable<MatrixBlock>
	{
		private MatrixBlock _in = null;
		private MatrixBlock _out = null;
		private boolean _row = false;
		private int _rl = -1;
		private int _ru = -1;
		private int[] _cnt = null;
		private boolean allowReturnBlock;

		protected TransposeTask(MatrixBlock in, MatrixBlock out, boolean row, int rl, int ru, int[] cnt, boolean returnBlock) {
			_in = in;
			_out = out;
			_row = row;
			_rl = rl;
			_ru = ru;
			_cnt = cnt;
			allowReturnBlock = returnBlock;
		}
		
		@Override
		public MatrixBlock call() {
			int rl = _row ? _rl : 0;
			int ru = _row ? _ru : _in.rlen;
			int cl = _row ? 0 : _rl;
			int cu = _row ? _in.clen : _ru;
			
			//execute transpose operation
			if( !_in.sparse && !_out.sparse )
				transposeDenseToDense( _in, _out, rl, ru, cl, cu );
			else if( _in.sparse && _out.sparse && _out.sparseBlock instanceof SparseBlockCSR)
				transposeSparseToSparseCSR(_in, _out, rl, ru, cl, cu, _cnt);
			else if( _in.sparse && _out.sparse && _in.isUltraSparse(false) )
				transposeUltraSparse(_in, _out, rl, ru, cl, cu);
			else if( _in.sparse && _out.sparse ){
				if(allowReturnBlock)
					return transposeSparseToSparseBlock(_in, rl, ru);
				
				transposeSparseToSparse( _in, _out, rl, ru, cl, cu, _cnt );
			}
			else if( _in.sparse )
				transposeSparseToDense( _in, _out, rl, ru, cl, cu );
			else
				throw new DMLRuntimeException("Unsupported multi-threaded dense-sparse transpose.");
			
			return null;
		}
	}

	private static class CountNnzTask implements Callable<int[]>
	{
		private MatrixBlock _in = null;
		private int _rl = -1;
		private int _ru = -1;

		protected CountNnzTask(MatrixBlock in, int rl, int ru) {
			_in = in;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public int[] call() {
			return countNnzPerColumn(_in, _rl, _ru);
		}
	}
	
	private static class RExpandColsTask implements Callable<Long>
	{
		private final MatrixBlock _in;
		private final MatrixBlock _out;
		private final int _max;
		private final boolean _cast;
		private final boolean _ignore;
		private final int _rl;
		private final int _ru;

		protected RExpandColsTask(MatrixBlock in, MatrixBlock out, int max, boolean cast, boolean ignore, int rl, int ru) {
			_in = in;
			_out = out;
			_max = max;
			_cast = cast;
			_ignore = ignore;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() {
			return rexpandColumns(_in, _out, _max, _cast, _ignore, _rl, _ru);
		}
	}

	private static class SortTask implements Callable<Object>
	{
		private final int _start;
		private final int _end;
		private final int[] _indexes;
		private final double[] _values;


		protected SortTask(int start, int end, int[] indexes, double[] values){
			_start = start;
			_end = end;
			_indexes = indexes;
			_values = values;
			
		}

		@Override
		public Long call(){
			SortUtils.sortByValue(_start, _end, _values, _indexes);
			return 1l;
		}

	}

	private static class MergeTask implements Callable<Object>
	{
		private final int _start;
		private final int _end;
		private final int _blockSize;
		private final int[] _indexes;
		private final double[] _values;

		protected MergeTask(int start, int end, int blockSize, int[] indexes, double[] values){
			_start = start;
			_end = end;
			_blockSize = blockSize;
			_indexes = indexes;
			_values = values;
		}

		@Override
		public Long call(){
			// Find the middle index of the two merge blocks
			int middle = _start + _blockSize;
			// Early return if edge case
			if (middle == _end) return 1l;
			
			// Pointer left side merge.
			int pointlIndex = middle -1;
			// Pointer to merge index.
			int positionToAssign = _end - 1;
			
			// Make copy of entire right side so that we use left side directly as output.
			// Results in worst case (and most cases) allocation of extra array of half input size. 
			int[] rhsCopy = Arrays.copyOfRange(_indexes, middle, _end);
			double[] rhsCopyV = Arrays.copyOfRange(_values, middle, _end);
			
			int pointrIndex = _end - middle - 1;
			while (positionToAssign >= _start && pointrIndex >= 0) {
				if( pointrIndex < 0 
					|| ( pointlIndex >= _start && _values[pointlIndex] > rhsCopyV[pointrIndex]) )
				{
					_values[positionToAssign] = _values[pointlIndex];
					_indexes[positionToAssign] = _indexes[pointlIndex];
					pointlIndex--;
					positionToAssign--;
				}
				else {
					_values[positionToAssign] = rhsCopyV[pointrIndex];
					_indexes[positionToAssign] = rhsCopy[pointrIndex];
					positionToAssign--;
					pointrIndex--;
				}
			}

			return 1l;
		}
	}
	
	private static class CopyTask implements Callable<Object>
	{
		private final MatrixBlock _in;
		private final MatrixBlock _out;
		private final int[] _vix;
		private final int _rl;
		private final int _ru;

		protected CopyTask(MatrixBlock in, MatrixBlock out, int[] vix, int rl, int ru) {
			_in = in;
			_out = out;
			_vix = vix;
			_rl = rl;
			_ru = ru;
		}

		@Override
		public Object call() {
			int clen = _in.clen;
			if( !_out.sparse ) { //DENSE
				DenseBlock a = _in.getDenseBlock();
				DenseBlock c = _out.getDenseBlock();
				for( int i=_rl; i<_ru; i++ )
					System.arraycopy(a.values(_vix[i]), a.pos(_vix[i]), c.values(i), c.pos(i), clen);
			}
			else { //SPARSE
				for( int i=_rl; i<_ru; i++ )
					if( !_in.sparseBlock.isEmpty(_vix[i]) )
						_out.sparseBlock.set(i, _in.sparseBlock.get(_vix[i]),
							!SHALLOW_COPY_REORG); //row remains unchanged
			}
			return null;
		}
	}

	/**
	 * Transposes a dense matrix in-place using following cycles based on Brenner's method. This
	 * method shifts cycles with a focus on less storage by using cycle leaders based on prime factorization. The used
	 * storage is in O(n+m). Quadratic matrices should be handled outside this method (using the trivial method) for a
	 * speedup. This method is based on: Algorithm 467, Brenner, https://dl.acm.org/doi/pdf/10.1145/355611.362542.
	 *
	 * @param in The input matrix to be transposed.
	 * @param k  The number of threads.
	 */
	public static void transposeInPlaceDenseBrenner(MatrixBlock in, int k) {

		DenseBlock denseBlock = in.getDenseBlock();
		double[] matrix = in.getDenseBlockValues();

		final int rows = in.getNumRows();
		final int cols = in.getNumColumns();

		// Brenner: rows + cols / 2 is sufficient for most cases.
		int workSize = rows + cols;
		int maxIndex = rows * cols - 1;

		// prime factorization of the maximum index to identify cycle structures
		// Brenner: length 8 is sufficient up to maxIndex 2*3*5*...*19 = 9,767,520.
		int[] primes = new int[8];
		int[] exponents = new int[8];
		int[] powers = new int[8];
		int numPrimes = primeFactorization(maxIndex, primes, exponents, powers);

		int[] iExponents = new int[numPrimes];
		int div = 1;

		div:
		while(div < maxIndex / 2) {

			// number of indices divisible by div and no other divisor of maxIndex
			int count = eulerTotient(primes, exponents, iExponents, numPrimes, maxIndex / div);
			// all false
			boolean[] moved = new boolean[workSize];
			// starting point cycle
			int start = div;

			count:
			do {
				// companion of start
				int comp = maxIndex - start;

				if(start == div) {
					// shift cycles
					count = simultaneousCycleShift(matrix, moved, rows, maxIndex, count, workSize, start, comp);
					start += div;
				}
				else if(start < workSize && moved[start]) {
					// already moved
					start += div;
				}
				else {
					// handle other cycle starts
					int cycleLeader = start / div;
					for(int ip = 0; ip < numPrimes; ip++) {
						if(iExponents[ip] != exponents[ip] && cycleLeader % primes[ip] == 0) {
							start += div;
							continue count;
						}
					}

					if(start < workSize) {
						count = simultaneousCycleShift(matrix, moved, rows, maxIndex, count, workSize, start, comp);
						start += div;
						continue;
					}

					int test = start;
					do {
						test = prevIndexCycle(test, rows, cols);
						if(test < start || test > comp) {
							start += div;
							continue count;
						}
					}
					while(test > start && test < comp);

					count = simultaneousCycleShift(matrix, moved, rows, maxIndex, count, workSize, start, comp);
					start += div;
				}
			}
			while(count > 0);

			// update cycle divisor for the next set of cycles based on prime factors
			for(int ip = 0; ip < numPrimes; ip++) {
				if(iExponents[ip] != exponents[ip]) {
					iExponents[ip]++;
					div *= primes[ip];
					continue div;
				}
				iExponents[ip] = 0;
				div /= powers[ip];
			}
		}

		denseBlock.setDims(new int[] {cols, rows});
		in.setNumColumns(rows);
		in.setNumRows(cols);
	}

	/**
	 * Performs a simultaneous cycle shift for a cycle and its companion cycle. This method ensures that distinct cycles
	 * or self-dual cycles are handled correctly. This method is based on: Algorithm 2, Karlsson,
	 * https://webapps.cs.umu.se/uminf/reports/2009/011/part1.pdf and Algorithm 467, Brenner,
	 * https://dl.acm.org/doi/pdf/10.1145/355611.362542.
	 *
	 * @param matrix   The matrix whose elements are being shifted.
	 * @param moved    Boolean array tracking whether an element has already been moved.
	 * @param rows     The number of rows in the matrix.
	 * @param maxIndex The maximum valid index in the matrix.
	 * @param count    The number of elements left to process.
	 * @param workSize The length of moved.
	 * @param start    The starting index for the cycle shift.
	 * @param comp     The corresponding companion index.
	 * @return The updated count of elements remaining to shift.
	 */
	private static int simultaneousCycleShift(double[] matrix, boolean[] moved, int rows, int maxIndex, int count,
		int workSize, int start, int comp) {

		int orig = start;
		double val = matrix[orig];
		double cval = matrix[comp];

		while(orig >= 0) {
			// decrease the remaining shift count by orig and comp
			count -= 2;
			orig = simultaneousCycleShiftStep(matrix, moved, rows, maxIndex, workSize, start, orig, val, cval);
		}
		return count;
	}

	private static int simultaneousCycleShiftStep(double[] matrix, boolean[] moved, int rows, int maxIndex,
		int workSize, int start, int orig, double val, double cval) {

		int comp = maxIndex - orig;
		int prevOrig = prevIndexCycle(orig, rows, (maxIndex + 1) / rows);
		int prevComp = maxIndex - prevOrig;

		if(orig < workSize)
			moved[orig] = true;
		if(comp < workSize)
			moved[comp] = true;

		if(prevOrig == start) {
			// cycle and comp are distinct
			matrix[orig] = val;
			matrix[comp] = cval;
			return -1;
		}
		else if(prevComp == start) {
			// cycle is self dual
			matrix[orig] = cval;
			matrix[comp] = val;
			return -1;
		}

		// shift the values to their next positions
		matrix[orig] = matrix[prevOrig];
		matrix[comp] = matrix[prevComp];
		// update
		return prevOrig;
	}

	private static int prevIndexCycle(int index, int rows, int cols) {
		int lastIndex = rows * cols - 1;
		if(index == lastIndex)
			return lastIndex;
		long temp = (long) index * cols;
		return (int) (temp % lastIndex);
	}

	/**
	 * Performs prime factorization of a given number n. The method calculates the prime factors of n, their exponents,
	 * powers and stores the results in the provided arrays.
	 *
	 * @param n         The number to be factorized.
	 * @param primes    Array to store the unique prime factors of n.
	 * @param exponents Array to store the exponents of the respective prime factors.
	 * @param powers    Array to store the powers of the respective prime factors.
	 * @return The number of unique prime factors.
	 */
	private static int primeFactorization(int n, int[] primes, int[] exponents, int[] powers) {
		int pIdx = -1;
		int currDiv = 0;
		int rest = n;
		int div = 2;

		while(rest > 1) {
			int quotient = rest / div;
			if(rest - div * quotient == 0) {
				if(div == currDiv) {
					// current divisor is the same as the last one
					powers[pIdx] *= div;
					exponents[pIdx]++;
				}
				else {
					// new prime factor found
					pIdx++;
					if(pIdx >= primes.length)
						throw new RuntimeException("Not enough space, need to expand input arrays.");

					primes[pIdx] = div;
					powers[pIdx] = div;
					currDiv = div;
					exponents[pIdx] = 1;
				}
				rest = quotient;
			}
			else {
				// only odd divs
				div = (div == 2) ? 3 : div + 2;
			}
		}
		return pIdx + 1;
	}

	private static int eulerTotient(int[] primes, int[] exponents, int[] iExponents, int numPrimes, int count) {
		for(int ip = 0; ip < numPrimes; ip++) {
			if(iExponents[ip] == exponents[ip]) {
				continue;
			}
			count = (count / primes[ip]) * (primes[ip] - 1);
		}
		return count;
	}
}
