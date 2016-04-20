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


package org.apache.sysml.runtime.instructions.spark.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

import org.apache.sysml.lops.Checkpoint;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBinaryCellFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.util.UtilFunctions;

public class SparkUtils 
{	
	//internal configuration
	public static final StorageLevel DEFAULT_TMP = Checkpoint.DEFAULT_STORAGE_LEVEL;
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static IndexedMatrixValue toIndexedMatrixBlock( Tuple2<MatrixIndexes,MatrixBlock> in ) {
		return new IndexedMatrixValue(in._1(), in._2());
	}
	
	/**
	 * 
	 * @param ix
	 * @param mb
	 * @return
	 */
	public static IndexedMatrixValue toIndexedMatrixBlock( MatrixIndexes ix, MatrixBlock mb ) {
		return new IndexedMatrixValue(ix, mb);
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static Tuple2<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlock( IndexedMatrixValue in ){
		return new Tuple2<MatrixIndexes,MatrixBlock>(in.getIndexes(), (MatrixBlock)in.getValue());
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlock( ArrayList<IndexedMatrixValue> in )
	{
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		for( IndexedMatrixValue imv : in )
			ret.add(fromIndexedMatrixBlock(imv));
		
		return ret;
	}
	
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static Tuple2<Long,FrameBlock> fromIndexedFrameBlock( Pair<Long, FrameBlock> in ){
		return new Tuple2<Long, FrameBlock>(in.getKey(), (FrameBlock)in.getValue());
	}
	
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static ArrayList<Tuple2<Long,FrameBlock>> fromIndexedFrameBlock( ArrayList<Pair<Long, FrameBlock>> in )
	{
		ArrayList<Tuple2<Long, FrameBlock>> ret = new ArrayList<Tuple2<Long, FrameBlock>>();
		for( Pair<Long, FrameBlock> ifv : in )
			ret.add(fromIndexedFrameBlock(ifv));
		
		return ret;
	}
	
	
	/**
	 * 
	 * @param mb
	 * @param blen
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock[] partitionIntoRowBlocks( MatrixBlock mb, int blen ) 
		throws DMLRuntimeException
	{
		//in-memory rowblock partitioning (according to bclen of rdd)
		int lrlen = mb.getNumRows();
		int numBlocks = (int)Math.ceil((double)lrlen/blen);				
		MatrixBlock[] partBlocks = new MatrixBlock[numBlocks];
		for( int i=0; i<numBlocks; i++ )
		{
			MatrixBlock tmp = new MatrixBlock();
			mb.sliceOperations(i*blen, Math.min((i+1)*blen, lrlen)-1, 
					0, mb.getNumColumns()-1, tmp);
			partBlocks[i] = tmp;
		}			
		
		return partBlocks;
	}
	
	/**
	 * 
	 * @param mb
	 * @param brlen
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static MatrixBlock[] partitionIntoColumnBlocks( MatrixBlock mb, int blen ) 
		throws DMLRuntimeException
	{
		//in-memory colblock partitioning (according to brlen of rdd)
		int lclen = mb.getNumColumns();
		int numBlocks = (int)Math.ceil((double)lclen/blen);				
		MatrixBlock[] partBlocks = new MatrixBlock[numBlocks];
		for( int i=0; i<numBlocks; i++ )
		{
			MatrixBlock tmp = new MatrixBlock();
			mb.sliceOperations(0, mb.getNumRows()-1, 
					i*blen, Math.min((i+1)*blen, lclen)-1,  tmp);
			partBlocks[i] = tmp;
		}
		
		return partBlocks;
	}
	
	// This returns RDD with identifier as well as location
	public static String getStartLineFromSparkDebugInfo(String line) throws DMLRuntimeException {
		// To remove: (2)  -- Assumption: At max, 9 RDDs as input to transformation/action
		String withoutPrefix = line.substring(4, line.length());
		// To remove: [Disk Memory Deserialized 1x Replicated]
		return  withoutPrefix.split(":")[0]; // Return 'MapPartitionsRDD[51] at mapToPair at ReorgSPInstruction.java'
	}
	
	public static String getPrefixFromSparkDebugInfo(String line) {
		String [] lines = line.split("\\||\\+-");
		String retVal = lines[0];
		for(int i = 1; i < lines.length-1; i++) {
			retVal += "|" + lines[i];
		}
		String twoSpaces = "  ";
		if(line.contains("+-"))
			return retVal + "+- ";
		else
			return retVal + "|" + twoSpaces;
	}
			
	
	// len = {clen or rlen}, blen = {brlen or bclen}
	public static long getStartGlobalIndex(long blockIndex, int blen, long len) {
		return UtilFunctions.computeCellIndex(blockIndex, blen, 0);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDDWithEmptyBlocks(JavaSparkContext sc, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithoutEmptyBlocks,
			long numRows, long numColumns, int brlen, int bclen) throws DMLRuntimeException {
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithEmptyBlocks = null;
		// ----------------------------------------------------------------------------
		// Now take care of empty blocks
		// This is done as non-rdd operation due to complexity involved in "not in" operations
		// Since this deals only with keys and not blocks, it might not be that bad.
		List<MatrixIndexes> indexes = binaryBlocksWithoutEmptyBlocks.keys().collect();
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock> > emptyBlocksList = getEmptyBlocks(indexes, numRows, numColumns, brlen, bclen);
		if(emptyBlocksList != null && emptyBlocksList.size() > 0) {
			// Empty blocks needs to be inserted
			binaryBlocksWithEmptyBlocks = JavaPairRDD.fromJavaRDD(sc.parallelize(emptyBlocksList))
					.union(binaryBlocksWithoutEmptyBlocks);
		}
		else {
			binaryBlocksWithEmptyBlocks = binaryBlocksWithoutEmptyBlocks;
		}
		// ----------------------------------------------------------------------------
		return binaryBlocksWithEmptyBlocks;
	}
	
	private static ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> getEmptyBlocks(List<MatrixIndexes> nonEmptyIndexes, long rlen, long clen, int brlen, int bclen) throws DMLRuntimeException {
		long numBlocksPerRow = (long) Math.ceil((double)rlen / brlen);
		long numBlocksPerCol = (long) Math.ceil((double)clen / bclen);
		long expectedNumBlocks = numBlocksPerRow*numBlocksPerCol;
		
		if(expectedNumBlocks == nonEmptyIndexes.size()) {
			return null; // no empty blocks required: sanity check
		}
		else if(expectedNumBlocks < nonEmptyIndexes.size()) {
			throw new DMLRuntimeException("Error: Incorrect number of indexes in ReblockSPInstruction:" + nonEmptyIndexes.size());
		}
		
		// ----------------------------------------------------------------------------
		// Add empty blocks: Performs a "not-in" operation
		Collections.sort(nonEmptyIndexes); // sort in ascending order first wrt rows and then wrt columns
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int index = 0;
		for(long row = 1; row <=  Math.ceil((double)rlen / brlen); row++) {
			for(long col = 1; col <=  Math.ceil((double)clen / bclen); col++) {
				boolean matrixBlockExists = false;
				if(nonEmptyIndexes.size() > index) {
					matrixBlockExists = (nonEmptyIndexes.get(index).getRowIndex() == row) && (nonEmptyIndexes.get(index).getColumnIndex() == col);
				}
				if(matrixBlockExists) {
					index++; // No need to add empty block
				}
				else {
					// ------------------------------------------------------------------
					//	Compute local block size: 
					// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
					// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
					long blockRowIndex = row;
					long blockColIndex = col;
					int emptyBlk_lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
					int emptyBlk_lclen = UtilFunctions.computeBlockSize(clen, blockColIndex, bclen);
					// ------------------------------------------------------------------
					
					MatrixBlock emptyBlk = new MatrixBlock(emptyBlk_lrlen, emptyBlk_lclen, true);
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(blockRowIndex, blockColIndex), emptyBlk));
				}
			}
		}
		// ----------------------------------------------------------------------------
		
		if(index != nonEmptyIndexes.size()) {
			throw new DMLRuntimeException("Unexpected error while adding empty blocks");
		}
		
		return retVal;
	}
	
	/**
	 * 
	 * @param sc
	 * @param mc
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getEmptyBlockRDD( JavaSparkContext sc, MatrixCharacteristics mc )
	{
		//create all empty blocks
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> list = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int nrblks = (int)Math.ceil((double)mc.getRows()/mc.getRowsPerBlock());
		int ncblks = (int)Math.ceil((double)mc.getCols()/mc.getColsPerBlock());
		for(long r=1; r<=nrblks; r++)
			for(long c=1; c<=ncblks; c++)
			{
				int lrlen = UtilFunctions.computeBlockSize(mc.getRows(), r, mc.getRowsPerBlock());
				int lclen = UtilFunctions.computeBlockSize(mc.getCols(), c, mc.getColsPerBlock());
				MatrixIndexes ix = new MatrixIndexes(r, c);
				MatrixBlock mb = new MatrixBlock(lrlen, lclen, true);
				list.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
			}
		
		//create rdd of in-memory list
		return sc.parallelizePairs(list);
	}
	
	/**
	 * 
	 * @param input
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixCell> cacheBinaryCellRDD(JavaPairRDD<MatrixIndexes, MatrixCell> input)
	{
		JavaPairRDD<MatrixIndexes, MatrixCell> ret = null;
		
		if( !input.getStorageLevel().equals(DEFAULT_TMP) ) {
			ret = input.mapToPair(new CopyBinaryCellFunction())
					   .persist(DEFAULT_TMP);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param input
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> cacheBinaryBlockRDD(JavaPairRDD<MatrixIndexes, MatrixBlock> input)
	{
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = null;
		
		if( !input.getStorageLevel().equals(DEFAULT_TMP) ) {
			ret = input.mapValues(new CopyBlockFunction(false))
					   .persist(DEFAULT_TMP);
		}
		
		return ret;
	}
	
	/**
	 * Utility to compute dimensions and non-zeros in a given RDD of binary cells.
	 * 
	 * @param rdd
	 * @param computeNNZ
	 * @return
	 */
	public static MatrixCharacteristics computeMatrixCharacteristics(JavaPairRDD<MatrixIndexes, MatrixCell> input) 
	{
		// compute dimensions and nnz in single pass
		MatrixCharacteristics ret = input
				.map(new AnalyzeCellMatrixCharacteristics())
				.reduce(new AggregateMatrixCharacteristics());
		
		return ret;
	}
	
	/**
	 * Utility to compute dimensions and non-zeros in the given RDD of matrix blocks.
	 * 
	 * @param rdd
	 * @param rpb
	 * @param cpb
	 * @param computeNNZ
	 * @return
	 */
	public static MatrixCharacteristics computeMatrixCharacteristics(JavaPairRDD<MatrixIndexes, MatrixBlock> input, int brlen, int bclen) 
	{
		// compute dimensions and nnz in single pass
		MatrixCharacteristics ret = input
				.map(new AnalyzeBlockMatrixCharacteristics(brlen, bclen))
				.reduce(new AggregateMatrixCharacteristics());
		
		return ret;
	}
	
	/**
	 * 
	 */
	private static class AnalyzeCellMatrixCharacteristics implements Function<Tuple2<MatrixIndexes,MatrixCell>, MatrixCharacteristics> 
	{
		private static final long serialVersionUID = 8899395272683723008L;

		@Override
		public MatrixCharacteristics call(Tuple2<MatrixIndexes, MatrixCell> arg0) 
			throws Exception 
		{
			long rix = arg0._1().getRowIndex();
			long cix = arg0._1().getColumnIndex();
			long nnz = (arg0._2().getValue()!=0) ? 1 : 0;
			return new MatrixCharacteristics(rix, cix, 0, 0, nnz);
		}
	}
	
	/**
	 * 
	 */
	private static class AnalyzeBlockMatrixCharacteristics implements Function<Tuple2<MatrixIndexes,MatrixBlock>, MatrixCharacteristics> 
	{
		private static final long serialVersionUID = -1857049501217936951L;
		
		private int _brlen = -1; 
		private int _bclen = -1; 
		
		public AnalyzeBlockMatrixCharacteristics(int brlen, int bclen) {
			_brlen = brlen;
			_bclen = bclen;
		}
		
		@Override
		public MatrixCharacteristics call(Tuple2<MatrixIndexes, MatrixBlock> arg0) 
			throws Exception 
		{
			MatrixBlock block = arg0._2();
			long rlen = (arg0._1().getRowIndex()-1)*_brlen + block.getNumRows();
			long clen = (arg0._1().getColumnIndex()-1)*_bclen + block.getNumColumns();
			long nnz = block.getNonZeros();
			return new MatrixCharacteristics(rlen, clen, _brlen, _bclen, nnz);
		}
	}
	
	/**
	 * 
	 */
	private static class AggregateMatrixCharacteristics implements Function2<MatrixCharacteristics, MatrixCharacteristics, MatrixCharacteristics> 
	{
		private static final long serialVersionUID = 4263886749699779994L;

		@Override
		public MatrixCharacteristics call(MatrixCharacteristics arg0, MatrixCharacteristics arg1) 
			throws Exception 
		{
			return new MatrixCharacteristics(
					Math.max(arg0.getRows(), arg1.getRows()),  //max
					Math.max(arg0.getCols(), arg1.getCols()),  //max
					arg0.getRowsPerBlock(), 
					arg0.getColsPerBlock(),
					arg0.getNonZeros() + arg1.getNonZeros() ); //sum
		}	
	}
	
	////////////////////////////
	//TODO MB: to be cleaned up but still used
	
	/**
	 * Utility to compute number of non-zeros from the given RDD of MatrixCells
	 * @param rdd
	 * @return
	 */
	public static long computeNNZFromCells(JavaPairRDD<MatrixIndexes, MatrixCell> rdd) {
		long nnz = rdd.values().filter(
						new Function<MatrixCell,Boolean>() {
							private static final long serialVersionUID = -6550193680630537857L;
							@Override
							public Boolean call(MatrixCell v1) throws Exception {
								return (v1.getValue() != 0);
							}
						}).count();
		return nnz;
	}
	
	/**
	 * Utility to compute number of non-zeros from the given RDD of MatrixBlocks
	 * @param rdd
	 * @return
	 */
	public static long computeNNZFromBlocks(JavaPairRDD<MatrixIndexes, MatrixBlock> rdd) {
		long nnz = rdd.values().aggregate(	0L, 
						new Function2<Long,MatrixBlock,Long>() {
							private static final long serialVersionUID = 4907645080949985267L;
							@Override
							public Long call(Long v1, MatrixBlock v2) throws Exception {
								return (v1 + v2.getNonZeros());
							} 
						}, 
						new Function2<Long,Long,Long>() {
							private static final long serialVersionUID = 333028431986883739L;
							@Override
							public Long call(Long v1, Long v2) throws Exception {
								return v1+v2;
							}
						} );
		return nnz;
	}
}
