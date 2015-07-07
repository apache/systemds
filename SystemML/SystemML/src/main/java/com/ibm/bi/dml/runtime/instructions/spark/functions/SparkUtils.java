/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class SparkUtils {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @param mb
	 * @param blen
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public static MatrixBlock[] partitionIntoRowBlocks( MatrixBlock mb, int blen ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//in-memory rowblock partitioning (according to bclen of rdd)
		int lrlen = mb.getNumRows();
		int numBlocks = (int)Math.ceil((double)lrlen/blen);				
		MatrixBlock[] partBlocks = new MatrixBlock[numBlocks];
		for( int i=0; i<numBlocks; i++ )
		{
			MatrixBlock tmp = new MatrixBlock();
			mb.sliceOperations(i*blen+1, Math.min((i+1)*blen, lrlen), 
					1, mb.getNumColumns(), tmp);
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
	 * @throws DMLUnsupportedOperationException
	 */
	public static MatrixBlock[] partitionIntoColumnBlocks( MatrixBlock mb, int blen ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//in-memory colblock partitioning (according to brlen of rdd)
		int lclen = mb.getNumColumns();
		int numBlocks = (int)Math.ceil((double)lclen/blen);				
		MatrixBlock[] partBlocks = new MatrixBlock[numBlocks];
		for( int i=0; i<numBlocks; i++ )
		{
			MatrixBlock tmp = new MatrixBlock();
			mb.sliceOperations(1, mb.getNumRows(), 
					i*blen+1, Math.min((i+1)*blen, lclen),  tmp);
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
		return UtilFunctions.cellIndexCalculation(blockIndex, blen, 0);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDDWithEmptyBlocks(SparkExecutionContext sec, 
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
			binaryBlocksWithEmptyBlocks = JavaPairRDD.fromJavaRDD(sec.getSparkContext().parallelize(emptyBlocksList))
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
}
