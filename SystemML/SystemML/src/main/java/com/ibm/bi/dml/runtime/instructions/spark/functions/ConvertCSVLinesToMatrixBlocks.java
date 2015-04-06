package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.google.common.base.Splitter;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class ConvertCSVLinesToMatrixBlocks implements Function2<Integer, Iterator<String>, Iterator<Tuple2<MatrixIndexes, MatrixBlock>>> {

	private static final long serialVersionUID = -2045829891472792200L;
	
	private int brlen; private int bclen;
	private String delim;
	private boolean fill;
	private double missingValue;
	private boolean hasHeader; 
	
	private Broadcast<HashMap<Integer, Long>> lineMap;
	public ConvertCSVLinesToMatrixBlocks(Broadcast<HashMap<Integer, Long>> offsetsBroadcast, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double missingValue) {
		this.lineMap = offsetsBroadcast;
		this.brlen = brlen;
		this.bclen = bclen;
		this.hasHeader = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.missingValue = missingValue;
	}

	@Override
	public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Integer partNo, final Iterator<String> lines) throws Exception {
		
		// Look up the offset of the first line.
		final long firstLineNum = lineMap.value().get(partNo);
		
		// Generate an iterator object that walks through the lines of the partition,
        // generating chunks on demand
		return new Iterator<Tuple2<MatrixIndexes,MatrixBlock>>() {
			
			// Cached contents of the current line, or null if we've reached end of line
			private String curLine = null;
			
			// How much of the current line we have consumed, in characters and cells
	        // of the matrix
			private int curCharOffset = -1;
			private int curCellOffset = -1;
			
			// Global offset of the current line within the original CSV file
	        // Initialized to 1 before the beginning of the partition to simplify 
	        // logic below
			private long curLineNum = firstLineNum - 1;
			
			@Override
			public boolean hasNext() {
				return (lines.hasNext() || (null != curLine));
			}

			@Override
			public Tuple2<MatrixIndexes, MatrixBlock> next() {
				// Read and parse the next line if we have no line buffered.
	            if (null == curLine) {
	              curLine = lines.next();
	              curCharOffset = 0;
	              curCellOffset = 0;
	              curLineNum += 1;
	            }
	            
	            // Collect up the next block's worth of values.
	            long blockColIx = curCellOffset / bclen;
	            long blockRowIx = curLineNum / brlen;
	            long rowOffsetWithinBlock = curLineNum % brlen;
	            
	            MatrixBlock curChunk = new MatrixBlock(brlen, bclen, true);
	            int colOffsetWithinChunk = 0;
	            
	            // TODO: Take care of missing values and simplify this logic using Guava Splitter
	            while (null != curLine && curCellOffset < (blockColIx + 1) * bclen) {
	                int nextCommaOffset = curLine.indexOf(',', curCharOffset);
	                double curCellVal = 0.0;
	                if (-1 == nextCommaOffset) {
	                  // End of line
	                  curCellVal = Double.parseDouble(curLine.substring(curCharOffset));
	                  curCharOffset = -1;
	                  curLine = null;
	                } else {
	                  curCellVal = Double.parseDouble(curLine.substring(curCharOffset, nextCommaOffset));
	                  curCharOffset = nextCommaOffset + 1;
	                }

	                // Use the special method for adding a value that isn't already 
	                // in the sparse matrix block
	                curChunk.appendValue((int) rowOffsetWithinBlock, colOffsetWithinChunk,
	                  curCellVal);
	                //curChunk(curOffsetInChunk) = curCellVal
	                curCellOffset += 1;
	                colOffsetWithinChunk += 1;
	              }
	            
	            // To support 1-based indexing
	            return new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(blockRowIx+1, blockColIx+1), curChunk);
			}

			@Override
			public void remove() {
				// TODO:
				// throw new Exception("Cannot remove entries from the iterator in CSVReblockSPInstruction");
			}

			
		};
		
	}
	
	
}