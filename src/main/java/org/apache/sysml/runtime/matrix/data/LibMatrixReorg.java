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


package org.apache.sysml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.functionobjects.DiagIndex;
import org.apache.sysml.runtime.functionobjects.RevIndex;
import org.apache.sysml.runtime.functionobjects.SortIndex;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.SortUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

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
public class LibMatrixReorg 
{
	public static final long PAR_NUMCELL_THRESHOLD = 1024*1024;   //Min 1M elements
	public static final boolean SHALLOW_DENSE_VECTOR_TRANSPOSE = true;
	public static final boolean SHALLOW_DENSE_ROWWISE_RESHAPE = true;
	public static final boolean ALLOW_BLOCK_REUSE = false;
	
	private enum ReorgType {
		TRANSPOSE,
		REV,
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
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public static boolean isSupportedReorgOperator( ReorgOperator op )
	{
		return (getReorgType(op) != ReorgType.INVALID);
	}

	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock reorg( MatrixBlock in, MatrixBlock out, ReorgOperator op ) 
		throws DMLRuntimeException
	{
		ReorgType type = getReorgType(op);
		
		switch( type )
		{
			case TRANSPOSE: 
				if( op.getNumThreads() > 1 )
					return transpose(in, out, op.getNumThreads());
				else
					return transpose(in, out);
			case REV: 
				return rev(in, out);
			case DIAG:      
				return diag(in, out); 
			case SORT:      
				SortIndex ix = (SortIndex) op.fn;
				return sort(in, out, ix.getCol(), ix.getDecreasing(), ix.getIndexReturn());
			
			default:        
				throw new DMLRuntimeException("Unsupported reorg operator: "+op.fn);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock transpose( MatrixBlock in, MatrixBlock out ) 
		throws DMLRuntimeException
	{
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return out;
	
		//set basic meta data
		out.nonZeros = in.nonZeros;
		
		//shallow dense vector transpose (w/o result allocation)
		//since the physical representation of dense vectors is always the same,
		//we don't need to create a copy, given our copy on write semantics.
		//however, note that with update in-place this would be an invalid optimization
		if( SHALLOW_DENSE_VECTOR_TRANSPOSE && !in.sparse && !out.sparse && (in.rlen==1 || in.clen==1)  ) {
			out.denseBlock = in.denseBlock;
			return out;
		}
		
		//Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		if( out.sparse )
			out.allocateSparseRowsBlock(false);
		else
			out.allocateDenseBlock(false);
	
		//execute transpose operation
		if( !in.sparse && !out.sparse )
			transposeDenseToDense( in, out, 0, in.rlen, 0, in.clen );
		else if( in.sparse && out.sparse )
			transposeSparseToSparse( in, out );
		else if( in.sparse )
			transposeSparseToDense( in, out, 0, in.rlen, 0, in.clen );
		else
			transposeDenseToSparse( in, out );
		
		//System.out.println("r' ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param k
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock transpose( MatrixBlock in, MatrixBlock out, int k ) 
		throws DMLRuntimeException
	{
		//redirect small or special cases to sequential execution
		if( in.isEmptyBlock(false) || (in.rlen * in.clen < PAR_NUMCELL_THRESHOLD)
			|| (SHALLOW_DENSE_VECTOR_TRANSPOSE && !in.sparse && !out.sparse && (in.rlen==1 || in.clen==1) )
			|| (in.sparse && !out.sparse && in.rlen==1) || out.sparse || !out.isThreadSafe())
		{
			return transpose(in, out);
		}
		
		//Timing time = new Timing(true);
		
		//set meta data and allocate output arrays (if required)
		out.nonZeros = in.nonZeros;
		if( out.sparse )
			out.allocateSparseRowsBlock(false);
		else
			out.allocateDenseBlock(false);
		
		//core multi-threaded transpose
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<TransposeTask> tasks = new ArrayList<TransposeTask>();
			boolean row = in.sparse || in.rlen >= in.clen;
			int len = row ? in.rlen : in.clen;
			int blklen = (int)(Math.ceil((double)len/k));
			blklen += (blklen%8 != 0)?8-blklen%8:0;
			for( int i=0; i<k & i*blklen<in.rlen; i++ )
				tasks.add(new TransposeTask(in, out, row, i*blklen, Math.min((i+1)*blklen, len)));
			//execute tasks and check for errors
			List<Future<Object>> taskret = pool.invokeAll(tasks);	
			pool.shutdown();
			for( Future<Object> task : taskret )
				task.get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}	
		
		//System.out.println("r' k="+k+" ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock rev( MatrixBlock in, MatrixBlock out ) 
		throws DMLRuntimeException
	{
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
	
	/**
	 * 
	 * @param in
	 * @param rows1
	 * @param brlen
	 * @param out
	 * @throws DMLRuntimeException  
	 */
	public static void rev( IndexedMatrixValue in, long rlen, int brlen, ArrayList<IndexedMatrixValue> out ) 
		throws DMLRuntimeException
	{
		//input block reverse 
		MatrixIndexes inix = in.getIndexes();
		MatrixBlock inblk = (MatrixBlock) in.getValue(); 
		MatrixBlock tmpblk = rev(inblk, new MatrixBlock(inblk.getNumRows(), inblk.getNumColumns(), inblk.isInSparseFormat()));
		
		//split and expand block if necessary (at most 2 blocks)
		if( rlen % brlen == 0 ) //special case: aligned blocks 
		{
			int nrblks = (int)Math.ceil((double)rlen/brlen);
			out.add(new IndexedMatrixValue(
					new MatrixIndexes(nrblks-inix.getRowIndex()+1, inix.getColumnIndex()), tmpblk));
		}
		else //general case: unaligned blocks
		{
			//compute target positions and sizes
			long pos1 = rlen - UtilFunctions.computeCellIndex(inix.getRowIndex(), brlen, tmpblk.getNumRows()-1) + 1;
			long pos2 = pos1 + tmpblk.getNumRows() - 1;
			int ipos1 = UtilFunctions.computeCellInBlock(pos1, brlen);
			int iposCut = tmpblk.getNumRows() - ipos1 - 1;
			int blkix1 = (int)UtilFunctions.computeBlockIndex(pos1, brlen);
			int blkix2 = (int)UtilFunctions.computeBlockIndex(pos2, brlen);
			int blklen1 = (int)UtilFunctions.computeBlockSize(rlen, blkix1, brlen);
			int blklen2 = (int)UtilFunctions.computeBlockSize(rlen, blkix2, brlen);
			
			//slice first block
			MatrixIndexes outix1 = new MatrixIndexes(blkix1, inix.getColumnIndex());
			MatrixBlock outblk1 = new MatrixBlock(blklen1, inblk.getNumColumns(), inblk.isInSparseFormat());
			MatrixBlock tmp1 = tmpblk.sliceOperations(0, iposCut, 0, tmpblk.getNumColumns()-1, new MatrixBlock());
			outblk1.leftIndexingOperations(tmp1, ipos1, outblk1.getNumRows()-1, 0, tmpblk.getNumColumns()-1, outblk1, UpdateType.INPLACE_PINNED);
			out.add(new IndexedMatrixValue(outix1, outblk1));
			
			//slice second block (if necessary)
			if( blkix1 != blkix2 ) {
				MatrixIndexes outix2 = new MatrixIndexes(blkix2, inix.getColumnIndex());
				MatrixBlock outblk2 = new MatrixBlock(blklen2, inblk.getNumColumns(), inblk.isInSparseFormat());
				MatrixBlock tmp2 = tmpblk.sliceOperations(iposCut+1, tmpblk.getNumRows()-1, 0, tmpblk.getNumColumns()-1, new MatrixBlock());
				outblk2.leftIndexingOperations(tmp2, 0, tmp2.getNumRows()-1, 0, tmpblk.getNumColumns()-1, outblk2, UpdateType.INPLACE_PINNED);
				out.add(new IndexedMatrixValue(outix2, outblk2));		
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock diag( MatrixBlock in, MatrixBlock out ) 
		throws DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return out;
		
		int rlen = in.rlen;
		int clen = in.clen;
		
		if( clen == 1 ){ //diagV2M
			diagV2M( in, out );
			out.setDiag();
		}
		else if ( rlen == clen ) //diagM2V
			diagM2V( in, out );
		else
			throw new DMLRuntimeException("Reorg diagM2V requires squared block input. ("+rlen+", "+clen+")");
		
		//System.out.println("rdiag ("+in.rlen+", "+in.clen+", "+in.sparse+", "+out.sparse+") in "+time.stop()+" ms.");
		
		return out;
	}
	
	/**
	 * 
	 * 
	 * @param in
	 * @param out
	 * @param by
	 * @param desc
	 * @param ixret
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock sort(MatrixBlock in, MatrixBlock out, int by, boolean desc, boolean ixret) 
		throws DMLRuntimeException
	{
		//meta data gathering and preparation
		boolean sparse = in.isInSparseFormat();
		int rlen = in.rlen;
		int clen = in.clen;
		out.sparse = (in.sparse && !ixret);
		out.nonZeros = ixret ? rlen : in.nonZeros;
		
		//step 1: error handling
		if( by <= 0 || clen < by )
			throw new DMLRuntimeException("Sort configuration issue: non-existing orderby column: "+by+" ("+rlen+"x"+clen+" input).");
		
		//step 2: empty block / special case handling
		if( !ixret ) //SORT DATA
		{
			if( in.isEmptyBlock(false) ) //EMPTY INPUT BLOCK
				return out;
			
			if( !sparse && clen == 1 ) { //DENSE COLUMN VECTOR
				//in-place quicksort, unstable (no indexes needed)
				out.copy( in ); //dense
				Arrays.sort(out.denseBlock);
				if( desc )
					sortReverseDense(out);
				return out;
			}
		}
		else //SORT INDEX
		{
			if( in.isEmptyBlock(false) ) { //EMPTY INPUT BLOCK
				out.allocateDenseBlock(false);
				for( int i=0; i<rlen; i++ ) //seq(1,n)
					out.setValueDenseUnsafe(i, 0, i+1);
				return out;
			}
		}
		
		//step 3: index vector sorting
		
		//create index vector and extract values
		int[] vix = new int[rlen];
		double[] values = new double[rlen];
		for( int i=0; i<rlen; i++ ) {
			vix[i] = i;
			values[i] = in.quickGetValue(i, by-1);
		}
		
		//sort index vector on extracted data (unstable)
		SortUtils.sortByValue(0, rlen, values, vix);

		//flip order if descending requested (note that this needs to happen
		//before we ensure stable outputs, hence we also flip values)
		if(desc) {
			sortReverseDense(vix);
			sortReverseDense(values);
		}
		
		//final pass to ensure stable output
		for( int i=0; i<rlen-1; i++ ) {
			double tmp = values[i];
			//determine run of equal values
			int len = 0;
			while( i+len+1<rlen && tmp==values[i+len+1] )
				len++;
			//unstable sort of run indexes (equal value guaranteed)
			if( len>0 ) {
				Arrays.sort(vix, i, i+len+1);
				i += len; //skip processed run
			}
		}

		//step 4: create output matrix (guaranteed non-empty, see step 2)
		if( !ixret )
		{
			//copy input data in sorted order into result
			if( !sparse ) //DENSE
			{
				out.allocateDenseBlock(false);
				for( int i=0; i<rlen; i++ ) {
					System.arraycopy(in.denseBlock, vix[i]*clen, out.denseBlock, i*clen, clen);
				}
			}
			else //SPARSE
			{
				out.allocateSparseRowsBlock(false);
				for( int i=0; i<rlen; i++ ) {
					int ix = vix[i];
					if( !in.sparseBlock.isEmpty(ix) ) {
						out.sparseBlock.set(i, in.sparseBlock.get(ix), true);
					}
				}
			}
		}
		else
		{
			//copy sorted index vector into result
			out.allocateDenseBlock(false);
			for( int i=0; i<rlen; i++ )
				out.setValueDenseUnsafe(i, 0, vix[i]+1);
		}
		
		return out;
	}
	
	/**
	 * CP reshape operation (single input, single output matrix) 
	 *
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock reshape( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise ) 
		throws DMLRuntimeException
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//check validity
		if( ((long)rlen)*clen != ((long)rows)*cols )
			throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells ("+rlen+":"+clen+", "+rows+":"+cols+").");
		
		//check for same dimensions
		if( rlen==rows && clen == cols ) {
			out.copy(in); //incl dims, nnz
			return out;
		}
	
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
			reshapeSparse(in, out, rows, cols, rowwise);
		else if(in.sparse)
			reshapeSparseToDense(in, out, rows, cols, rowwise);
		else
			reshapeDenseToSparse(in, out, rows, cols, rowwise);
		
		return out;
	}


	/**
	 * MR reshape interface - for reshape we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 *  
	 * @param in
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param out
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ArrayList<IndexedMatrixValue> reshape( IndexedMatrixValue in, long rows1, long cols1, int brlen1, int bclen1, 
			                      ArrayList<IndexedMatrixValue> out, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise ) 	
		throws DMLRuntimeException
	{
		//prepare inputs
		MatrixIndexes ixIn = in.getIndexes();
		MatrixBlock mbIn = (MatrixBlock) in.getValue();
		
		//prepare result blocks (no reuse in order to guarantee mem constraints)
		Collection<MatrixIndexes> rix = computeAllResultBlockIndexes(ixIn, rows1, cols1, brlen1, bclen1, rows2, cols2, brlen2, bclen2, rowwise);
		HashMap<MatrixIndexes, MatrixBlock> rblk = createAllResultBlocks(rix, mbIn.nonZeros, rows1, cols1, brlen1, bclen1, rows2, cols2, brlen2, bclen2, rowwise, out);
		
		//basic algorithm
		long row_offset = (ixIn.getRowIndex()-1)*brlen1;
		long col_offset = (ixIn.getColumnIndex()-1)*bclen1;
		if( mbIn.sparse )
			reshapeSparse(mbIn, row_offset, col_offset, rblk, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
		else //dense
			reshapeDense(mbIn, row_offset, col_offset, rblk, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);

		//prepare output
		out = new ArrayList<IndexedMatrixValue>();
		for( Entry<MatrixIndexes, MatrixBlock> e : rblk.entrySet() )
			out.add(new IndexedMatrixValue(e.getKey(),e.getValue()));
		
		return out;
	}
	
	/**
	 * CP rmempty operation (single input, single output matrix) 
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock rmempty(MatrixBlock in, MatrixBlock ret, boolean rows) 
		throws DMLRuntimeException
	{
		return rmempty(in, ret, rows, null);
	}
		
	/**
	 * CP rmempty operation (single input, single output matrix) 
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @throws DMLRuntimeException 
	 */
	public static MatrixBlock rmempty(MatrixBlock in, MatrixBlock ret, boolean rows, MatrixBlock select) 
		throws DMLRuntimeException
	{
		//check for empty inputs 
		//(the semantics of removeEmpty are that for an empty m-by-n matrix, the output 
		//is an empty 1-by-n or m-by-1 matrix because we don't allow matrices with dims 0)
		if( in.isEmptyBlock(false) ) {
			if( rows )
				ret.reset(1, in.clen, in.sparse);
			else //cols
				ret.reset(in.rlen, 1, in.sparse);	
			return ret;
		}
		
		if( rows )
			return removeEmptyRows(in, ret, select);
		else //cols
			return removeEmptyColumns(in, ret, select);
	}

	/**
	 * MR rmempty interface - for rmempty we cannot view blocks independently, and hence,
	 * there are different CP and MR interfaces.
	 * 
	 * @param imv1
	 * @param imv2
	 * @param out
	 * @throws DMLRuntimeException 
	 */
	public static void rmempty(IndexedMatrixValue data, IndexedMatrixValue offset, boolean rmRows, long len, long brlen, long bclen, ArrayList<IndexedMatrixValue> outList) 
		throws DMLRuntimeException
	{
		//sanity check inputs
		if( !(data.getValue() instanceof MatrixBlock && offset.getValue() instanceof MatrixBlock) )
			throw new DMLRuntimeException("Unsupported input data: expected "+MatrixBlock.class.getName()+" but got "+data.getValue().getClass().getName()+" and "+offset.getValue().getClass().getName());
		if(     rmRows && data.getValue().getNumRows()!=offset.getValue().getNumRows() 
			|| !rmRows && data.getValue().getNumColumns()!=offset.getValue().getNumColumns()  ){
			throw new DMLRuntimeException("Dimension mismatch between input data and offsets: ["
					+data.getValue().getNumRows()+"x"+data.getValue().getNumColumns()+" vs "+offset.getValue().getNumRows()+"x"+offset.getValue().getNumColumns());
		}
		
		//compute outputs (at most two output blocks)
		HashMap<MatrixIndexes,IndexedMatrixValue> out = new HashMap<MatrixIndexes,IndexedMatrixValue>();
		MatrixBlock linData = (MatrixBlock) data.getValue();
		MatrixBlock linOffset = (MatrixBlock) offset.getValue();
		MatrixIndexes tmpIx = new MatrixIndexes(-1,-1);
		if( rmRows ) //margin = "rows"
		{
			long rlen = len;
			long clen = linData.getNumColumns();
			
			for( int i=0; i<linOffset.getNumRows(); i++ ) {
				long rix = (long)linOffset.quickGetValue(i, 0);
				if( rix > 0 ) //otherwise empty row
				{
					//get single row from source block
					MatrixBlock src = (MatrixBlock) linData.sliceOperations(
							  i, i, 0, (int)(clen-1), new MatrixBlock());
					long brix = (rix-1)/brlen+1;
					long lbrix = (rix-1)%brlen;
					tmpIx.setIndexes(brix, data.getIndexes().getColumnIndex());
					 //create target block if necessary
					if( !out.containsKey(tmpIx) ) {
						IndexedMatrixValue tmpIMV = new IndexedMatrixValue(new MatrixIndexes(),new MatrixBlock());
						tmpIMV.getIndexes().setIndexes(tmpIx);
						((MatrixBlock)tmpIMV.getValue()).reset((int)Math.min(brlen, rlen-((brix-1)*brlen)), (int)clen);
						out.put(tmpIMV.getIndexes(), tmpIMV);
					}
					//put single row into target block
					((MatrixBlock)out.get(tmpIx).getValue()).copy(
							  (int)lbrix, (int)lbrix, 0, (int)clen-1, src, false);
				}
			}
		}
		else //margin = "cols"
		{
			long rlen = linData.getNumRows();
			long clen = len;
			
			for( int i=0; i<linOffset.getNumColumns(); i++ ) {
				long cix = (long)linOffset.quickGetValue(0, i);
				if( cix > 0 ) //otherwise empty row
				{
					//get single row from source block
					MatrixBlock src = (MatrixBlock) linData.sliceOperations(
							  0, (int)(rlen-1), i, i, new MatrixBlock());
					long bcix = (cix-1)/bclen+1;
					long lbcix = (cix-1)%bclen;
					tmpIx.setIndexes(data.getIndexes().getRowIndex(), bcix);
					 //create target block if necessary
					if( !out.containsKey(tmpIx) ) {
						IndexedMatrixValue tmpIMV = new IndexedMatrixValue(new MatrixIndexes(),new MatrixBlock());
						tmpIMV.getIndexes().setIndexes(tmpIx);
						((MatrixBlock)tmpIMV.getValue()).reset((int)rlen,(int)Math.min(bclen, clen-((bcix-1)*bclen)));
						out.put(tmpIMV.getIndexes(), tmpIMV);
					}
					//put single row into target block
					((MatrixBlock)out.get(tmpIx).getValue()).copy(
							  0, (int)rlen-1, (int)lbcix, (int)lbcix, src, false);
				}
			}
		}
		
		//prepare and return outputs (already in cached values)
		for( IndexedMatrixValue imv : out.values() ){
			((MatrixBlock)imv.getValue()).recomputeNonZeros();
			outList.add(imv);
		}
	}

	/**
	 * CP rexpand operation (single input, single output)
	 * 
	 * @param in
	 * @param ret
	 * @param rows
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock rexpand(MatrixBlock in, MatrixBlock ret, double max, boolean rows, boolean cast, boolean ignore) 
		throws DMLRuntimeException
	{
		//prepare parameters
		int lmax = (int)UtilFunctions.toLong(max);
		
		//sanity check for input nnz (incl implicit handling of empty blocks)
		if( !ignore && in.getNonZeros()<in.getNumRows() )
			throw new DMLRuntimeException("Invalid input w/ zeros for rexpand ignore=false "
					+ "(rlen="+in.getNumRows()+", nnz="+in.getNonZeros()+").");
		
		//check for empty inputs (for ignore=true)
		if( in.isEmptyBlock(false) ) {
			if( rows )
				ret.reset(lmax, in.rlen, true);
			else //cols
				ret.reset(in.rlen, lmax, true);	
			return ret;
		}
		
		//execute rexpand operations
		if( rows )
			return rexpandRows(in, ret, lmax, cast, ignore);
		else //cols
			return rexpandColumns(in, ret, lmax, cast, ignore);
	}

	/**
	 * MR/Spark rexpand operation (single input, multiple outputs incl empty blocks)
	 * 
	 * @param data
	 * @param offset
	 * @param rmRows
	 * @param len
	 * @param brlen
	 * @param bclen
	 * @param outList
	 * @throws DMLRuntimeException
	 */
	public static void rexpand(IndexedMatrixValue data, double max, boolean rows, boolean cast, boolean ignore, long brlen, long bclen, ArrayList<IndexedMatrixValue> outList) 
		throws DMLRuntimeException
	{
		//prepare parameters
		MatrixIndexes ix = data.getIndexes();
		MatrixBlock in = (MatrixBlock)data.getValue();
		
		//execute rexpand operations incl sanity checks
		//TODO more robust (memory efficient) implementation w/o tmp block
		MatrixBlock tmp = rexpand(in, new MatrixBlock(), max, rows, cast, ignore);
		
		//prepare outputs blocks (slice tmp block into output blocks ) 
		if( rows ) //expanded vertically
		{
			for( int rl=0; rl<tmp.getNumRows(); rl+=brlen ) {
				MatrixBlock mb = tmp.sliceOperations(
						rl, (int)(Math.min(rl+brlen, tmp.getNumRows())-1), 
						0, tmp.getNumColumns()-1, new MatrixBlock());
				outList.add(new IndexedMatrixValue(
						new MatrixIndexes(rl/brlen+1, ix.getRowIndex()), mb));
			}
		}
		else //expanded horizontally
		{
			for( int cl=0; cl<tmp.getNumColumns(); cl+=bclen ) {
				MatrixBlock mb = tmp.sliceOperations(
						0, tmp.getNumRows()-1,
						cl, (int)(Math.min(cl+bclen, tmp.getNumColumns())-1), new MatrixBlock());
				outList.add(new IndexedMatrixValue(
						new MatrixIndexes(ix.getRowIndex(), cl/bclen+1), mb));
			}
		}
	}

	///////////////////////////////
	// private CP implementation //
	///////////////////////////////

	
	/**
	 * 
	 * @param op
	 * @return
	 */
	private static ReorgType getReorgType( ReorgOperator op )
	{
		if( op.fn instanceof SwapIndex )  //transpose
			return ReorgType.TRANSPOSE;
		
		else if( op.fn instanceof RevIndex ) //rev
			return ReorgType.REV;
		
		else if( op.fn instanceof DiagIndex ) //diag
			return ReorgType.DIAG;
		
		else if( op.fn instanceof SortIndex ) //sort
			return ReorgType.SORT;
				
		return ReorgType.INVALID;
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @throws DMLRuntimeException 
	 */
	private static void transposeDenseToDense(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu) 
		throws DMLRuntimeException
	{
		final int m = in.rlen;
		final int n = in.clen;
		final int n2 = out.clen;
		
		double[] a = in.getDenseBlock();
		double[] c = out.getDenseBlock();
		
		if( m==1 || n==1 ) //VECTOR TRANSPOSE
		{
			//plain memcopy, in case shallow dense copy no applied 
			int ix = rl+cl; int len = ru+cu-ix-1;
			System.arraycopy(a, ix, c, ix, len);
		}
		else //MATRIX TRANSPOSE
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128; 
			
			//blocked execution
			for( int bi = rl; bi<ru; bi+=blocksizeI )
				for( int bj = cl; bj<cu; bj+=blocksizeJ )
				{
					int bimin = Math.min(bi+blocksizeI, ru);
					int bjmin = Math.min(bj+blocksizeJ, cu);
					//core transpose operation
					for( int i=bi; i<bimin; i++ )
					{
						int aix = i * n + bj;
						int cix = bj * n2 + i;
						transposeRow(a, c, aix, cix, n2, bjmin-bj);
					}
				}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeDenseToSparse(MatrixBlock in, MatrixBlock out)
	{
		//NOTE: called only in sequential execution
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = (int) (in.nonZeros/m2); 
		
		double[] a = in.getDenseBlock();
		SparseBlock c = out.getSparseBlock();
		
		if( out.rlen == 1 ) //VECTOR-VECTOR
		{	
			c.allocate(0, (int)in.nonZeros); 
			c.setIndexRange(0, 0, m, a, 0, m);
		}
		else //general case: MATRIX-MATRIX
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128; 
			
			//blocked execution
			for( int bi = 0; bi<m; bi+=blocksizeI )
				for( int bj = 0; bj<n; bj+=blocksizeJ )
				{
					int bimin = Math.min(bi+blocksizeI, m);
					int bjmin = Math.min(bj+blocksizeJ, n);
					//core transpose operation
					for( int i=bi; i<bimin; i++ )				
						for( int j=bj, aix=i*n+bj; j<bjmin; j++, aix++ )
						{
							c.allocate(j, ennz2, n2); 
							c.append(j, i, a[aix]);
						}
				}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 */
	private static void transposeSparseToSparse(MatrixBlock in, MatrixBlock out)
	{
		//NOTE: called only in sequential execution
		
		final int m = in.rlen;
		final int n = in.clen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int ennz2 = (int) (in.nonZeros/m2); 
		
		SparseBlock a = in.getSparseBlock();
		SparseBlock c = out.getSparseBlock();

		//initial pass to determine capacity (this helps to prevent
		//sparse row reallocations and mem inefficiency w/ skew
		int[] cnt = null;
		if( n <= 4096 ) { //16KB
			cnt = new int[n];
			for( int i=0; i<m; i++ ) {
				if( !a.isEmpty(i) )
					countAgg(cnt, a.indexes(i), a.pos(i), a.size(i));
			}
		}
		
		//allocate output sparse rows
		if( cnt != null ) {
			for( int i=0; i<m2; i++ )
				if( cnt[i] > 0 )
					c.allocate(i, cnt[i]);
		}
		
		//blocking according to typical L2 cache sizes 
		final int blocksizeI = 128;
		final int blocksizeJ = 128; 
	
		//temporary array for block boundaries (for preventing binary search) 
		int[] ix = new int[blocksizeI];
		
		//blocked execution
		for( int bi = 0; bi<m; bi+=blocksizeI )
		{
			Arrays.fill(ix, 0);
			for( int bj = 0; bj<n; bj+=blocksizeJ )
			{
				int bimin = Math.min(bi+blocksizeI, m);
				int bjmin = Math.min(bj+blocksizeJ, n);

				//core transpose operation
				for( int i=bi, iix=0; i<bimin; i++, iix++ )
				{
					if( !a.isEmpty(i) ) {
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						int j = ix[iix]; //last block boundary
						for( ; j<alen && aix[j]<bjmin; j++ ) {
							c.allocate(aix[apos+j], ennz2,n2);
							c.append(aix[apos+j], i, avals[apos+j]);
						}
						ix[iix] = j; //keep block boundary
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @throws DMLRuntimeException 
	 */
	private static void transposeSparseToDense(MatrixBlock in, MatrixBlock out, int rl, int ru, int cl, int cu) 
		throws DMLRuntimeException
	{
		final int m = in.rlen;
		final int n = in.clen;
		final int n2 = out.clen;
		
		SparseBlock a = in.getSparseBlock();
		double[] c = out.getDenseBlock();
		
		if( m==1 ) //ROW VECTOR TRANSPOSE
		{
			//NOTE: called only in sequential execution
			int alen = a.size(0); //always pos 0
			int[] aix = a.indexes(0);
			double[] avals = a.values(0);
			for( int j=0; j<alen; j++ )
				c[ aix[j] ] = avals[j];
		}
		else //MATRIX TRANSPOSE
		{
			//blocking according to typical L2 cache sizes 
			final int blocksizeI = 128;
			final int blocksizeJ = 128; 
		
			//temporary array for block boundaries (for preventing binary search) 
			int[] ix = new int[blocksizeI];
			
			//blocked execution
			for( int bi = rl; bi<ru; bi+=blocksizeI )
			{
				Arrays.fill(ix, 0);
				for( int bj = 0; bj<n; bj+=blocksizeJ )
				{
					int bimin = Math.min(bi+blocksizeI, ru);
					int bjmin = Math.min(bj+blocksizeJ, n);
	
					//core transpose operation
					for( int i=bi, iix=0; i<bimin; i++, iix++ )
					{
						if( !a.isEmpty(i) ) {
							int apos = a.pos(i);
							int alen = a.size(i);
							int[] aix = a.indexes(i);
							double[] avals = a.values(i);
							int j = ix[iix]; //last block boundary
							for( ; j<alen && aix[apos+j]<bjmin; j++ )
								c[ aix[apos+j]*n2+i ] = avals[ apos+j ];
							ix[iix] = j; //keep block boundary						
						}
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param a
	 * @param c
	 * @param aix
	 * @param cix
	 * @param n2
	 * @param len
	 */
	private static void transposeRow( double[] a, double[] c, int aix, int cix, int n2, int len )
	{
		final int bn = len%8;
		
		//compute rest (not aligned to 8-blocks)
		for( int j=0; j<bn; j++, aix++, cix+=n2 )
			c[ cix ] = a[ aix+0 ];	
		
		//unrolled 8-blocks
		for( int j=bn; j<len; j+=8, aix+=8, cix+=8*n2 )
		{
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
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @throws DMLRuntimeException
	 */
	private static void reverseDense(MatrixBlock in, MatrixBlock out) 
		throws DMLRuntimeException
	{
		final int m = in.rlen;
		final int n = in.clen;
		final int len = m * n;
		
		//set basic meta data and allocate output
		out.sparse = false;
		out.nonZeros = in.nonZeros;
		out.allocateDenseBlock(false);
		
		double[] a = in.getDenseBlock();
		double[] c = out.getDenseBlock();
		
		//copy all rows into target positions
		if( n == 1 ) { //column vector
			for( int i=0; i<m; i++ )
				c[m-1-i] = a[i];
		}
		else { //general matrix case
			for( int i=0, aix=0; i<m; i++, aix+=n )
				System.arraycopy(a, aix, c, len-aix-n, n);
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @throws DMLRuntimeException
	 */
	private static void reverseSparse(MatrixBlock in, MatrixBlock out) 
		throws DMLRuntimeException
	{
		final int m = in.rlen;
		
		//set basic meta data and allocate output
		out.sparse = true;
		out.nonZeros = in.nonZeros;
		
		out.allocateSparseRowsBlock(false);
		
		SparseBlock a = in.getSparseBlock();
		SparseBlock c = out.getSparseBlock();
		
		//copy all rows into target positions
		for( int i=0; i<m; i++ ) {
			if( !a.isEmpty(i) ) {
				c.set(m-1-i, a.get(i), true);	
			}
		}
	}
	
	/**
	 * Generic implementation diagV2M (non-performance critical)
	 * (in most-likely DENSE, out most likely SPARSE)
	 * 
	 * @param in
	 * @param out
	 */
	private static void diagV2M( MatrixBlock in, MatrixBlock out )
	{
		int rlen = in.rlen;
		
		//CASE column vector
		for( int i=0; i<rlen; i++ )
		{
			double val = in.quickGetValue(i, 0);
			if( val != 0 )
				out.appendValue(i, i, val);
		}
	}
	
	/**
	 * Generic implementation diagM2V (non-performance critical)
	 * (in most-likely SPARSE, out most likely DENSE)
	 * 
	 * NOTE: squared block assumption (checked on entry diag)
	 * 
	 * @param in
	 * @param out
	 */
	private static void diagM2V( MatrixBlock in, MatrixBlock out )
	{
		int rlen = in.rlen;
		
		for( int i=0; i<rlen; i++ )
		{
			double val = in.quickGetValue(i, i);
			if( val != 0 )
				out.quickSetValue(i, 0, val);
		}
	}
	
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 * @throws DMLRuntimeException 
	 */
	private static void reshapeDense( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise ) 
		throws DMLRuntimeException
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.denseBlock == null )
			return;
		
		//shallow dense by-row reshape (w/o result allocation)
		if( SHALLOW_DENSE_ROWWISE_RESHAPE && rowwise ) {
			//since the physical representation of dense matrices is always the same,
			//we don't need to create a copy, given our copy on write semantics.
			//however, note that with update in-place this would be an invalid optimization
			out.denseBlock = in.denseBlock;
			return;
		}
		
		//allocate block if necessary
		out.allocateDenseBlock(false);
		
		//dense reshape
		double[] a = in.denseBlock;
		double[] c = out.denseBlock;
		
		if( rowwise )
		{
			//VECTOR-MATRIX, MATRIX-VECTOR, GENERAL CASE
			//pure copy of rowwise internal representation
			System.arraycopy(a, 0, c, 0, c.length);
		}	
		else //colwise
		{
			if( rlen==1 || clen==1 ) //VECTOR->MATRIX
			{
				//note: cache-friendly on a but not on c
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0, cix=0; i<rows; i++, cix+=cols )
						c[ cix + j ] = a[ aix++ ];
			}
			else if( rows==1 || cols==1 ) //MATRIX->VECTOR	
			{
				//note: cache-friendly on c but not on a
				for( int j=0, cix=0; j<clen; j++ )
					for( int i=0, aix=0; i<rlen; i++, aix+=clen )
						c[ cix++ ] = a[ aix + j ];
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on c but not an a
				for( int i=0, cix=0; i<rows; i++ )
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows )
					{
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						c[ cix++ ] = a[ ai*clen+aj ];				
					}			
				//index conversion c[i,j]<- a[k,l]: 
				// k = (rows*j+i)%rlen
				// l = (rows*j+i)/rlen
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
	private static void reshapeSparse( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise )
	{
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
				for( int i=0, cix=0; i<rlen; i++, cix+=clen ) 
				{
					if( !a.isEmpty(i) ) {
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);	
						for( int j=apos; j<apos+alen; j++ )
							c.append(0, cix+aix[j], avals[j]);
					}
				}
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c; append-only
				//long cix because total cells in sparse can be larger than int
				long cix = 0;
				
				for( int i=0; i<rlen; i++ ) 
				{
					if( !a.isEmpty(i) ){
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);	
						for( int j=apos; j<apos+alen; j++ )
						{
							int ci = (int)((cix+aix[j])/cols);
							int cj = (int)((cix+aix[j])%cols);       
							c.allocate(ci, estnnz, cols);
							c.append(ci, cj, avals[j]);
						}
					}	
					
					cix += clen;
				}
			}
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
					for( int j=0; j<alen; j++ ) 
					{
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
				for( int i=0; i<rlen; i++ ) 
				{
					if( !a.isEmpty(i) ){
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);	
						for( int j=apos; j<apos+alen; j++ )
						{
							//long tmpix because total cells in sparse can be larger than int
							long tmpix = (long)aix[j]*rlen+i;
							int ci = (int)(tmpix%rows);
							int cj = (int)(tmpix/rows); 
							c.allocate(ci, estnnz, cols);
							c.append(ci, cj, avals[j]);
						}
					}	
				}
				out.sortSparseRows();
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 */
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
		double[] a = in.denseBlock;
		SparseBlock c = out.sparseBlock;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c; append-only
			for( int i=0, aix=0; i<rows; i++ ) 
				for( int j=0; j<cols; j++ )
				{
					double val = a[aix++];
					if( val != 0 ) {
						c.allocate(i, estnnz, cols);
						c.append(i, j, val);
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
				for( int j=0, aix=0; j<cols; j++ )
					for( int i=0; i<rows; i++ ) 
					{
						double val = a[aix++];
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
					for( int j=0, aix2=i; j<cols; j++, aix2+=rows )
					{
						int ai = aix2%rlen;
						int aj = aix2/rlen;
						double val = a[ ai*clen+aj ];
						if( val != 0 ) {
							c.allocate(i, estnnz, cols);
							c.append(i, j, val);
						}
					}			
			}
		}
	}
	
	/**
	 * 
	 * @param in
	 * @param out
	 * @param rows
	 * @param cols
	 * @param rowwise
	 * @throws DMLRuntimeException 
	 */
	private static void reshapeSparseToDense( MatrixBlock in, MatrixBlock out, int rows, int cols, boolean rowwise ) 
		throws DMLRuntimeException
	{
		int rlen = in.rlen;
		int clen = in.clen;
		
		//reshape empty block
		if( in.sparseBlock == null )
			return;
		
		//allocate block if necessary
		out.allocateDenseBlock(false);
		
		//sparse/dense reshape
		SparseBlock a = in.sparseBlock;
		double[] c = out.denseBlock;
		
		if( rowwise )
		{
			//NOTES on special cases
			// * vector-matrix, matrix-vector not really different from general
			
			//GENERAL CASE: MATRIX->MATRIX
			//note: cache-friendly on a and c
			for( int i=0, cix=0; i<rlen; i++, cix+=clen ) 
			{
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);	
					for( int j=apos; j<apos+alen; j++ )
						c[cix+aix[j]] = avals[j];
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
				if( !a.isEmpty(0) ){
					int apos = a.pos(0);
					int alen = a.size(0);
					int[] aix = a.indexes(0);
					double[] avals = a.values(0);	
					for( int j=apos; j<apos+alen; j++ ) {
						int ci = aix[j]%rows;
						int cj = aix[j]/rows;       
						c[ci*cols+cj] = avals[j];
					}
				}								
			}
			else //GENERAL CASE: MATRIX->MATRIX
			{
				//note: cache-friendly on a but not c
				for( int i=0; i<rlen; i++ ) 
				{
					if( !a.isEmpty(i) ){
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);	
						for( int j=apos; j<apos+alen; j++ ) {
							int tmpix = aix[j]*rlen+i;
							int ci = tmpix%rows;
							int cj = tmpix/rows;   
							c[ci*cols+cj] = avals[j];
						}
					}	
				}
			}
		}
	}
	
	///////////////////////////////
	// private MR implementation //
	///////////////////////////////
	
	/**
	 * 
	 * @param ixin
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static Collection<MatrixIndexes> computeAllResultBlockIndexes( MatrixIndexes ixin,
            long rows1, long cols1, int brlen1, int bclen1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		HashSet<MatrixIndexes> ret = new HashSet<MatrixIndexes>();
		
		long nrblk2 = rows2/brlen2 + ((rows2%brlen2!=0)?1:0);
		long ncblk2 = cols2/bclen2 + ((cols2%bclen2!=0)?1:0);
		
		long row_offset = (ixin.getRowIndex()-1)*brlen1;
		long col_offset = (ixin.getColumnIndex()-1)*bclen1;
		
		if( rowwise ){
			for( long i=row_offset; i<Math.min(rows1,row_offset+brlen1); i++ )
			{
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), i, col_offset, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), i, Math.min(cols1,col_offset+bclen1)-1, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);

				if( first.getRowIndex()<=0 || first.getColumnIndex()<=0 )
					throw new RuntimeException("Invalid computed first index: "+first.toString());
				if( last.getRowIndex()<=0 || last.getColumnIndex()<=0 )
					throw new RuntimeException("Invalid computed last index: "+last.toString());
				
				//add first row block
				ret.add(first);
				
				//add blocks in between first and last
				for( long k1=first.getRowIndex(); k1<=last.getRowIndex(); k1++ )
				{
					long k2_start = ((k1==first.getRowIndex()) ? first.getColumnIndex()+1 : 1);
					long k2_end = ((k1==last.getRowIndex()) ? last.getColumnIndex()-1 : ncblk2);
					
					for( long k2=k2_start; k2<=k2_end; k2++ ) {
						ret.add(new MatrixIndexes(k1,k2));
					}
				}
				
				//add last row block
				ret.add(last);
				
			}
		}
		else{ //colwise
			for( long j=col_offset; j<Math.min(cols1,col_offset+bclen1); j++ )
			{
				MatrixIndexes first = computeResultBlockIndex(new MatrixIndexes(), row_offset, j, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
				MatrixIndexes last = computeResultBlockIndex(new MatrixIndexes(), Math.min(rows1,row_offset+brlen1)-1, j, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);

				if( first.getRowIndex()<=0 || first.getColumnIndex()<=0 )
					throw new RuntimeException("Invalid computed first index: "+first.toString());
				if( last.getRowIndex()<=0 || last.getColumnIndex()<=0 )
					throw new RuntimeException("Invalid computed last index: "+last.toString());
				
				//add first row block
				ret.add(first);
				//add blocks in between first and last
				for( long k1=first.getColumnIndex(); k1<=last.getColumnIndex(); k1++ )
				{
					long k2_start = ((k1==first.getColumnIndex()) ? first.getRowIndex()+1 : 1);
					long k2_end = ((k1==last.getColumnIndex()) ? last.getRowIndex()-1 : nrblk2);
					
					for( long k2=k2_start; k2<=k2_end; k2++ ) {
						ret.add(new MatrixIndexes(k1,k2));
					}
				}
				
				//add last row block
				ret.add(last);
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param brlen1
	 * @param bclen1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @param reuse 
	 * @return
	 */
	@SuppressWarnings("unused")
	private static HashMap<MatrixIndexes, MatrixBlock> createAllResultBlocks( Collection<MatrixIndexes> rix,
            long nnz, long rows1, long cols1, int brlen1, int bclen1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise, ArrayList<IndexedMatrixValue> reuse )
	{
		HashMap<MatrixIndexes, MatrixBlock> ret = new HashMap<MatrixIndexes,MatrixBlock>();
		long nBlocks = rix.size();
		int count = 0;
		
		//System.out.println("Reuse "+((reuse!=null)?reuse.size():0)+"/"+nBlocks);
		
		for( MatrixIndexes ix : rix )
		{
			//compute indexes
			long bi = ix.getRowIndex();
			long bj = ix.getColumnIndex();
			int lbrlen = (int) Math.min(brlen2, rows2-(bi-1)*brlen2);
			int lbclen = (int) Math.min(bclen2, cols2-(bj-1)*bclen2);
			
			//create result block
			int estnnz = (int) (nnz/nBlocks); //force initialcapacity per row to 1, for many blocks
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(lbrlen, lbclen, estnnz);
			MatrixBlock block = null;
			if( ALLOW_BLOCK_REUSE && reuse!=null && !reuse.isEmpty()) {
				block = (MatrixBlock) reuse.get(count++).getValue();
				block.reset(lbrlen, lbclen, sparse, estnnz);
			}
			else
				block = new MatrixBlock(lbrlen, lbclen, sparse, estnnz); 
			
			//System.out.println("create block ("+bi+","+bj+"): "+lbrlen+" "+lbclen);
			if( lbrlen<1 || lbclen<1 )
				throw new RuntimeException("Computed block dimensions ("+bi+","+bj+" -> "+lbrlen+","+lbclen+") are invalid!");
			
			ret.put(ix, block);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param row_offset
	 * @param col_offset
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 */
	private static void reshapeDense( MatrixBlock in, long row_offset, long col_offset, 
			HashMap<MatrixIndexes,MatrixBlock> rix,
            long rows1, long cols1, 
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
    {
		if( in.isEmptyBlock(false) )
			return;
		
		int rlen = in.rlen;
		int clen = in.clen;
		double[] a = in.denseBlock;
		
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
					computeResultBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					MatrixBlock out = rix.get(ixtmp);
					computeInBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), val);
				}
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise )
		{
			for( MatrixBlock block : rix.values() )
				if( block.sparse )
					block.sortSparseRows();
		}				
    }

	/**
	 * 
	 * @param in
	 * @param row_offset
	 * @param col_offset
	 * @param rix
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 */
	private static void reshapeSparse( MatrixBlock in, long row_offset, long col_offset, 
			HashMap<MatrixIndexes,MatrixBlock> rix,
            long rows1, long cols1,
            long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
    {
		if( in.isEmptyBlock(false) )
			return;
		
		int rlen = in.rlen;
		SparseBlock a = in.sparseBlock;
		
		//append all values to right blocks
		MatrixIndexes ixtmp = new MatrixIndexes();
		for( int i=0; i<rlen; i++ )
		{
			if( !a.isEmpty(i) ) {
				long ai = row_offset+i;
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				for( int j=apos; j<apos+alen; j++ )  {
					long aj = col_offset+aix[j];
					computeResultBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					MatrixBlock out = rix.get(ixtmp);
					computeInBlockIndex(ixtmp, ai, aj, rows1, cols1, rows2, cols2, brlen2, bclen2, rowwise);
					out.appendValue((int)ixtmp.getRowIndex(),(int)ixtmp.getColumnIndex(), avals[j]);
				}
			}
		}
		
		//cleanup for sparse blocks
		if( !rowwise )
		{
			for( MatrixBlock block : rix.values() )
				if( block.sparse )
					block.sortSparseRows();
		}				
    }
	
	/**
	 * Assumes internal (0-begin) indices ai, aj as input; computes external block indexes (1-begin) 
	 * 
	 * @param ixout
	 * @param ai
	 * @param aj
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static MatrixIndexes computeResultBlockIndex( MatrixIndexes ixout, long ai, long aj,
			            long rows1, long cols1, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		long ci, cj, tempc;
		long bci, bcj;
		
		if( rowwise ) {
			tempc = ai*cols1+aj;
			ci = tempc/cols2;
			cj = tempc%cols2;
			bci = ci/brlen2 + 1;
			bcj = cj/bclen2 + 1;
		}
		else { //colwise
			tempc = ai+rows1*aj;
			ci = tempc%rows2;
			cj = tempc/rows2;
			bci = ci/brlen2 + 1;
			bcj = cj/bclen2 + 1;			
		}
		
		ixout.setIndexes(bci, bcj);	
		return ixout;
	}
	
	/**
	 * 
	 * @param ixout
	 * @param ai
	 * @param aj
	 * @param rows1
	 * @param cols1
	 * @param rows2
	 * @param cols2
	 * @param brlen2
	 * @param bclen2
	 * @param rowwise
	 * @return
	 */
	private static MatrixIndexes computeInBlockIndex( MatrixIndexes ixout, long ai, long aj,
            long rows1, long cols1, long rows2, long cols2, int brlen2, int bclen2, boolean rowwise )
	{
		long ci, cj, tempc;
		
		if( rowwise ) {
			tempc = ai*cols1+aj;
			ci = (tempc/cols2)%brlen2;
			cj = (tempc%cols2)%bclen2;
		}
		else { //colwise
			tempc = ai+rows1*aj; 
			ci = (tempc%rows2)%brlen2;
			cj = (tempc/rows2)%bclen2;
		}
		
		//System.out.println("ai/aj: "+ai+"/"+aj+"  ->  ci/cj: "+ci+"/"+cj);

		ixout.setIndexes(ci, cj);	
		return ixout;
	}

	/**
	 * 
	 * @param in
	 * @param ret
	 * @param select
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock removeEmptyRows(MatrixBlock in, MatrixBlock ret, MatrixBlock select) 
		throws DMLRuntimeException 
	{	
		final int m = in.rlen;
		final int n = in.clen;
		boolean[] flags = null; 
		int rlen2 = 0; 
		
		if(select == null) 
		{
			flags = new boolean[ m ]; //false
			//Step 1: scan block and determine non-empty rows
			
			if( in.sparse ) //SPARSE 
			{
				SparseBlock a = in.sparseBlock;				
				for ( int i=0; i < m; i++ )
					if ( !a.isEmpty(i) ) {
						flags[i] = true;
						rlen2++;
					}
			}
			else //DENSE
			{
				double[] a = in.denseBlock;
				
				for(int i=0, aix=0; i<m; i++, aix+=n) {
					for(int j=0; j<n; j++)
						if( a[aix+j] != 0 )
						{
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
		rlen2 = Math.max(rlen2, 1); //ensure valid output
		boolean sp = MatrixBlock.evalSparseFormatInMemory(rlen2, n, in.nonZeros);
		ret.reset(rlen2, n, sp);
		
		if( in.sparse ) //* <- SPARSE
		{
			//note: output dense or sparse
			for( int i=0, cix=0; i<m; i++ )
				if( flags[i] )
					ret.appendRow(cix++, in.sparseBlock.get(i));
		}
		else if( !in.sparse && !ret.sparse )  //DENSE <- DENSE
		{
			ret.allocateDenseBlock();
			double[] a = in.denseBlock;
			double[] c = ret.denseBlock;
			
			for( int i=0, aix=0, cix=0; i<m; i++, aix+=n )
				if( flags[i] ) {
					System.arraycopy(a, aix, c, cix, n);
					cix += n; //target index
				}
		}
		else //SPARSE <- DENSE
		{
			ret.allocateSparseRowsBlock();
			double[] a = in.denseBlock;
			
			for( int i=0, aix=0, cix=0; i<m; i++, aix+=n )
				if( flags[i] ) {
					for( int j=0; j<n; j++ )
						ret.appendValue(cix, j, a[aix+j]);
					cix++;
				}
		}
		
		//check sparsity
		ret.nonZeros = in.nonZeros;
		ret.examSparsity();

		return ret;
	}

	
	/**
	 * @param in
	 * @param ret
	 * @param select
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock removeEmptyColumns(MatrixBlock in, MatrixBlock ret, MatrixBlock select) 
		throws DMLRuntimeException 
	{
		final int m = in.rlen;
		final int n = in.clen;
		
		//Step 1: scan block and determine non-empty columns 
		//(we optimized for cache-friendly behavior and hence don't do early abort)
		boolean[] flags = null; 
		
		if (select == null) 
		{
			flags = new boolean[ n ]; //false
			if( in.sparse ) //SPARSE 
			{
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
			else //DENSE
			{
				double[] a = in.denseBlock;
				
				for(int i=0, aix=0; i<m; i++)
					for(int j=0; j<n; j++, aix++)
						if( a[aix] != 0 )
							flags[j] = true; 	
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
		
		//Step 3: create mapping of flags to target indexes
		int[] cix = new int[n];
		for( int j=0, pos=0; j<n; j++ ) {
			if( flags[j] )
				cix[j] = pos++;	
		}
		
		//Step 3: reset result and copy cols
		//dense stays dense if correct input representation (but robust for any input), 
		// sparse might be dense/sparse
		clen2 = Math.max(clen2, 1); //ensure valid output
		boolean sp = MatrixBlock.evalSparseFormatInMemory(m, clen2, in.nonZeros);
		ret.reset(m, clen2, sp);
			
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
		else if( !in.sparse && !ret.sparse )  //DENSE <- DENSE
		{
			ret.allocateDenseBlock();
			double[] a = in.denseBlock;
			double[] c = ret.denseBlock;
			
			for(int i=0, aix=0, lcix=0; i<m; i++, lcix+=clen2)
				for(int j=0; j<n; j++, aix++)
					if( flags[j] )
						 c[ lcix+cix[j] ] = a[aix];	
		}
		else //SPARSE <- DENSE
		{
			ret.allocateSparseRowsBlock();
			double[] a = in.denseBlock;
			
			for(int i=0, aix=0; i<m; i++)
				for(int j=0; j<n; j++, aix++)
					if( flags[j] && a[aix]!=0 )
						 ret.appendValue(i, cix[j], a[aix]);	
		}
		
		//check sparsity
		ret.nonZeros = in.nonZeros;
		ret.examSparsity();
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param ret
	 * @param max
	 * @param cast
	 * @param ignore
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static MatrixBlock rexpandRows(MatrixBlock in, MatrixBlock ret, int max, boolean cast, boolean ignore) 
		throws DMLRuntimeException
	{
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
				if( val == Math.floor(val) && val >= 1 && val <= max )
					ret.appendValue((int)(val-1), i+tmpi[j], 1);
			}
			
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param ret
	 * @param max
	 * @param cast
	 * @param ignore
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static MatrixBlock rexpandColumns(MatrixBlock in, MatrixBlock ret, int max, boolean cast, boolean ignore) 
		throws DMLRuntimeException
	{
		//set meta data
		final int rlen = in.rlen;
		final int clen = max;
		final long nnz = in.nonZeros;
		boolean sp = MatrixBlock.evalSparseFormatInMemory(rlen, clen, nnz);
		ret.reset(rlen, clen, sp);
		
		//expand input horizontally (input vector likely dense 
		//but generic implementation for general case)
		for( int i=0; i<rlen; i++ )
		{
			//get value and cast if necessary (table)
			double val = in.quickGetValue(i, 0);
			if( cast )
				val = UtilFunctions.toLong(val);
			
			//handle invalid values if not to be ignored
			if( !ignore && val<=0 )
				throw new DMLRuntimeException("Invalid input value <= 0 for ignore=false: "+val);
				
			//set expanded value if matching
			if( val == Math.floor(val) && val >= 1 && val <= max )
				ret.appendValue(i, (int)(val-1), 1);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param in
	 * @param ixin
	 * @param tmp
	 * @param len
	 */
	private static void copyColVector( MatrixBlock in, int ixin, double[] tmp, int[] tmpi, int len)
	{
		//copy value array from input matrix
		if( in.isEmptyBlock(false) ) {
			Arrays.fill(tmp, 0, 0, len);
		}
		else if( in.sparse ){ //SPARSE
			for( int i=0; i<len; i++ )
				tmp[i] = in.quickGetValue(ixin+i, 0);
		}
		else { //DENSE
			System.arraycopy(in.denseBlock, ixin, tmp, 0, len);
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
	 * @param m1
	 */
	private static void sortReverseDense( MatrixBlock m1 )
	{
		int rlen = m1.rlen;
		double[] a = m1.denseBlock;
		
		for( int i=0; i<rlen/2; i++ ) {
			double tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}
	
	/**
	 * 
	 * @param m1
	 */
	private static void sortReverseDense( int[] a )
	{
		int rlen = a.length;
		
		for( int i=0; i<rlen/2; i++ ) {
			int tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}
	
	/**
	 * 
	 * @param a
	 */
	private static void sortReverseDense( double[] a )
	{
		int rlen = a.length;
		
		for( int i=0; i<rlen/2; i++ ) {
			double tmp = a[i];
			a[i] = a[rlen - i -1];
			a[rlen - i - 1] = tmp;
		}
	}

	/**
	 * 
	 * @param c
	 * @param ai
	 * @param len
	 */
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
	
	/**
	 *
	 */
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
		public int compare(Integer arg0, Integer arg1) 
		{			
			double val0 = _mb.quickGetValue(arg0, _col);
			double val1 = _mb.quickGetValue(arg1, _col);			
			return (val0 < val1 ? -1 : (val0 == val1 ? 0 : 1));
		}		
	}
	
	/**
	 * 
	 */
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
			double val0 = _mb.quickGetValue(arg0, _col);
			double val1 = _mb.quickGetValue(arg1, _col);	
			return (val0 > val1 ? -1 : (val0 == val1 ? 0 : 1));
		}		
	}
	
	/**
	 * 
	 */
	private static class TransposeTask implements Callable<Object>
	{
		private MatrixBlock _in = null;
		private MatrixBlock _out = null;
		private boolean _row = false;
		private int _rl = -1;
		private int _ru = -1;

		protected TransposeTask(MatrixBlock in, MatrixBlock out, boolean row, int rl, int ru) 
			throws DMLRuntimeException
		{
			_in = in;
			_out = out;
			_row = row;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() throws DMLRuntimeException
		{
			int rl = _row ? _rl : 0;
			int ru = _row ? _ru : _in.rlen;
			int cl = _row ? 0 : _rl;
			int cu = _row ? _in.clen : _ru;
			
			//execute transpose operation
			if( !_in.sparse && !_out.sparse )
				transposeDenseToDense( _in, _out, rl, ru, cl, cu );
			else if( _in.sparse && _out.sparse )
				throw new DMLRuntimeException("Unsupported multi-threaded sparse-sparse transpose.");
			else if( _in.sparse )
				transposeSparseToDense( _in, _out, rl, ru, cl, cu );
			else
				throw new DMLRuntimeException("Unsupported multi-threaded dense-sparse transpose.");
			
			return null;
		}
	}
}
