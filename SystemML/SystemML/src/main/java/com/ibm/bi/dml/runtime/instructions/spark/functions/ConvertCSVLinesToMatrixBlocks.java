package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.HashMap;
import java.util.Iterator;

import org.apache.spark.api.java.function.Function2;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ConvertCSVLinesToMatrixBlocks implements Function2<Integer, Iterator<String>, Iterator<Tuple2<MatrixIndexes, MatrixBlock>>> {

	private static final long serialVersionUID = -2045829891472792200L;
	
	private int brlen; private int bclen;
	private long rlen; private long clen; 
	private String delim;
	private boolean fill;
	private double missingValue;
	private boolean hasHeader; 
	
	private HashMap<Integer, Long> lineMap;
	public ConvertCSVLinesToMatrixBlocks(HashMap<Integer, Long> offsetsBroadcast, long rlen, long clen, int brlen, int bclen, boolean hasHeader, String delim, boolean fill, double missingValue) {
		this.lineMap = offsetsBroadcast;
		this.brlen = brlen;
		this.bclen = bclen;
		this.rlen = rlen;
		this.clen = clen;
		this.hasHeader = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.missingValue = missingValue;
	}

	@Override
	public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(final Integer partNo, final Iterator<String> lines) throws Exception {
		
		// Look up the offset of the first line.
		final long firstLineNum = lineMap.get(partNo);
		
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
			
			private double getValue(String entry) {
				if(entry.compareTo("") == 0) {
            		if(fill) {
            			return missingValue;
            		}
            		else {
            			// throw new Exception("Missing value in the line:" + curLine);
            			return Double.NaN;
            		}
            	}
            	else {
            		return Double.parseDouble(entry);
            	}
			}

			@Override
			public Tuple2<MatrixIndexes, MatrixBlock> next() {
				// Read and parse the next line if we have no line buffered.
	            if (null == curLine) {
	            	if(partNo == 0 && curLineNum == 0 && hasHeader) {
	            		lines.next(); // skip the header
	            	}
	            	
	              curLine = lines.next();
	              curCharOffset = 0;
	              curCellOffset = 0;
	              curLineNum += 1;
	            }
	            
	            // Collect up the next block's worth of values.
	            long blockColIx = curCellOffset / bclen;
	            long blockRowIx = curLineNum / brlen;
	            long rowOffsetWithinBlock = curLineNum % brlen;
	            
	            
	    		// ------------------------------------------------------------------
	    		//	Compute local block size: 
	    		// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
	    		// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
	    		long blockRowIndex = blockRowIx+1;
	    		long blockColIndex = blockColIx+1;
	    		int lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
	    		int lclen = UtilFunctions.computeBlockSize(clen, blockColIndex, bclen);
	    		// ------------------------------------------------------------------
	    		
	    		// Take care of non-square blocks
	    		MatrixBlock curChunk = new MatrixBlock(lrlen, lclen, true);
	            // MatrixBlock curChunk = new MatrixBlock(brlen, bclen, true);
	            
	            int colOffsetWithinChunk = 0;
	            
	            while (null != curLine && curCellOffset < (blockColIx + 1) * bclen) {
	                int nextCommaOffset = curLine.indexOf(delim, curCharOffset);
	                double curCellVal = 0.0;
	                if (-1 == nextCommaOffset) {
	                  // End of line
	                  curCellVal = getValue(curLine.substring(curCharOffset).trim());
	                  curCharOffset = -1;
	                  curLine = null;
	                } else {
	                  curCellVal = getValue(curLine.substring(curCharOffset, nextCommaOffset).trim());
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
			public void remove() { }

			
		};
		
	}
	
	
}