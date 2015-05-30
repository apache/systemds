package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.Explain;

public class SparkUtils {
	
	public static void setLineageInfoForExplain(Instruction inst, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> out, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1, String in1Name) throws DMLRuntimeException {
		setLineageInfoForExplain(inst, out, in1, in1Name, null, null);
	}
	
	// This returns RDD with identifier as well as location
	private static String getStartLineFromSparkDebugInfo(String line) throws DMLRuntimeException {
		// To remove: (2)  -- Assumption: At max, 9 RDDs as input to transformation/action
		String withoutPrefix = line.substring(4, line.length());
		// To remove: [Disk Memory Deserialized 1x Replicated]
		return  withoutPrefix.split(":")[0]; // Return 'MapPartitionsRDD[51] at mapToPair at ReorgSPInstruction.java'
	}
	
	private static String getPrefixFromSparkDebugInfo(String line) {
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
			
	// The most expensive operation here is rdd.toDebugString() which can be a major hit because
	// of unrolling lazy evaluation of Spark. Hence, it is guarded against it along with flag 'PRINT_EXPLAIN_WITH_LINEAGE' which is 
	// enabled only through MLContext. This way, it doesnot affect our performance evaluation through non-MLContext path
	public static void setLineageInfoForExplain(Instruction inst, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> out, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1, String in1Name, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in2, String in2Name) throws DMLRuntimeException {
		if(inst.getDebugString() == null && Explain.PRINT_EXPLAIN_WITH_LINEAGE) {
			// First fetch start lines from input RDDs
			String startLine1 = null; 
			String startLine2 = null;
			int i1length = 0, i2length = 0;
			if(in1 != null) {
				String [] lines = in1.toDebugString().split("\\r?\\n");
				startLine1 = getStartLineFromSparkDebugInfo(lines[0]); // lines[0].substring(4, lines[0].length());
				i1length = lines.length;
			}
			if(in2 != null) {
				String [] lines = in2.toDebugString().split("\\r?\\n");
				startLine2 =  getStartLineFromSparkDebugInfo(lines[0]); // lines[0].substring(4, lines[0].length());
				i2length = lines.length;
			}
			
			String outDebugString = "";
			int skip = 0;
			
			// Now process output RDD and replace inputRDD debug string by the matrix variable name
			String [] outLines = out.toDebugString().split("\\r?\\n");
			for(int i = 0; i < outLines.length; i++) {
				if(skip > 0) {
					skip--;
					// outDebugString += "\nSKIP:" + outLines[i];
				}
				else if(startLine1 != null && outLines[i].contains(startLine1)) {
					String prefix = getPrefixFromSparkDebugInfo(outLines[i]); // outLines[i].substring(0, outLines[i].length() - startLine1.length());
					outDebugString += "\n" + prefix + "[[" + in1Name + "]]";
					//outDebugString += "\n{" + prefix + "}[[" + in1Name + "]] => " + outLines[i];
					skip = i1length - 1;  
				}
				else if(startLine2 != null && outLines[i].contains(startLine2)) {
					String prefix = getPrefixFromSparkDebugInfo(outLines[i]); // outLines[i].substring(0, outLines[i].length() - startLine2.length());
					outDebugString += "\n" + prefix + "[[" + in2Name + "]]";
					skip = i2length - 1;
				}
				else {
					outDebugString += "\n" + outLines[i];
				}
			}
			
			// outDebugString += "\n{" + startLine1 + "}\n{" + startLine2 + "}";
			
			inst.setDebugString(outDebugString + "\n");
		}
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
