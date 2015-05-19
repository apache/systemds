package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ExtractBlockForBinaryReblock implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
	private static final long serialVersionUID = -762987655085029215L;
	long rlen; long clen; 
	int in_brlen; int in_bclen; int out_brlen; int out_bclen;
	public ExtractBlockForBinaryReblock(long rlen, long clen, int in_brlen, int in_bclen, int out_brlen, int out_bclen) {
		this.rlen = rlen; this.clen = clen;
		this.in_brlen = in_brlen; this.in_bclen = in_bclen;
		this.out_brlen = out_brlen; this.out_bclen = out_bclen;
	}
	
	private long getStartGlobalIndex(long blockIndex, boolean isIn, boolean isRow) {
		if(isIn && isRow) {
			return UtilFunctions.cellIndexCalculation(blockIndex, in_brlen, 0);
		}
		else if(isIn && !isRow) {
			return UtilFunctions.cellIndexCalculation(blockIndex, in_bclen, 0);
		}
		else if(!isIn && isRow) {
			return UtilFunctions.cellIndexCalculation(blockIndex, out_brlen, 0);
		}
		else {
			return UtilFunctions.cellIndexCalculation(blockIndex, out_bclen, 0);
		}
	}
	
	private long getEndGlobalIndex(long blockIndex, boolean isIn, boolean isRow) {
		if(isIn && isRow) {
			int new_lrlen = UtilFunctions.computeBlockSize(rlen, blockIndex, in_brlen);
			return UtilFunctions.cellIndexCalculation(blockIndex, in_brlen, new_lrlen-1);
		}
		else if(isIn && !isRow) {
			int new_lclen = UtilFunctions.computeBlockSize(clen, blockIndex, in_bclen);
			return UtilFunctions.cellIndexCalculation(blockIndex, in_bclen, new_lclen-1);
		}
		else if(!isIn && isRow) {
			int new_lrlen = UtilFunctions.computeBlockSize(rlen, blockIndex, out_brlen);
			return UtilFunctions.cellIndexCalculation(blockIndex, out_brlen, new_lrlen-1);
		}
		else {
			int new_lclen = UtilFunctions.computeBlockSize(clen, blockIndex, out_bclen);
			return UtilFunctions.cellIndexCalculation(blockIndex, out_bclen, new_lclen-1);
		}
	}
	
	private long getBlockIndex(long globalCellIndex, boolean isIn, boolean isRow) {
		if(isIn && isRow) {
			return UtilFunctions.blockIndexCalculation(globalCellIndex, in_brlen);
		}
		else if(isIn && !isRow) {
			return UtilFunctions.blockIndexCalculation(globalCellIndex, in_bclen);
		}
		else if(!isIn && isRow) {
			return UtilFunctions.blockIndexCalculation(globalCellIndex, out_brlen);
		}
		else {
			return UtilFunctions.blockIndexCalculation(globalCellIndex, out_bclen);
		}
	}
	

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		// The global cell indexes don't change in reblock operations
		long startRowGlobalCellIndex = getStartGlobalIndex(kv._1.getRowIndex(), true, true);
		long endRowGlobalCellIndex = getEndGlobalIndex(kv._1.getRowIndex(), true, true);
		long startColGlobalCellIndex = getStartGlobalIndex(kv._1.getColumnIndex(), true, false);
		long endColGlobalCellIndex = getEndGlobalIndex(kv._1.getColumnIndex(), true, false);
		
		if(startRowGlobalCellIndex > endRowGlobalCellIndex || startColGlobalCellIndex > endColGlobalCellIndex) {
			throw new Exception("Incorrect global cell calculation");
		}
		
		long out_startRowBlockIndex = getBlockIndex(startRowGlobalCellIndex, false, true);
		long out_endRowBlockIndex = getBlockIndex(endRowGlobalCellIndex, false, true);
		long out_startColBlockIndex = getBlockIndex(startColGlobalCellIndex, false, false);
		long out_endColBlockIndex = getBlockIndex(endColGlobalCellIndex, false, false);
		
		if(out_startRowBlockIndex > out_endRowBlockIndex || out_startColBlockIndex > out_endColBlockIndex) {
			throw new Exception("Incorrect block calculation");
		}
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
//		if(out_startRowBlockIndex == out_endRowBlockIndex && out_startColBlockIndex == out_endColBlockIndex) {
//			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(out_startRowBlockIndex, out_startColBlockIndex), kv._2));
//			return retVal;
//		}
		
		for(long i = out_startRowBlockIndex; i <= out_endRowBlockIndex; i++) {
			for(long j = out_startColBlockIndex; j <= out_endColBlockIndex; j++) {
				MatrixIndexes indx = new MatrixIndexes(i, j);
				long rowLower = Math.max(getStartGlobalIndex(i, false, true), startRowGlobalCellIndex);
				long rowUpper = Math.min(getEndGlobalIndex(i, false, true), endRowGlobalCellIndex);
				long colLower = Math.max(getStartGlobalIndex(j, false, false), startColGlobalCellIndex);
				long colUpper = Math.min(getEndGlobalIndex(j, false, false), endColGlobalCellIndex);
				
				int new_lrlen = UtilFunctions.computeBlockSize(rlen, i, out_brlen);
				int new_lclen = UtilFunctions.computeBlockSize(clen, j, out_bclen);
				MatrixBlock blk = new MatrixBlock(new_lrlen, new_lclen, true);
				
				for(long i1 = rowLower; i1 <= rowUpper; i1++) {
					SparseRow row = new SparseRow(new_lclen);
					int in_i1 = UtilFunctions.cellInBlockCalculation(i1, in_brlen);
					int out_i1 = UtilFunctions.cellInBlockCalculation(i1, out_brlen);
					boolean atleastOneNonZero = false;
					for(long j1 = colLower; j1 <= colUpper; j1++) {
						int in_j1 = UtilFunctions.cellInBlockCalculation(j1, in_bclen);
						int out_j1 = UtilFunctions.cellInBlockCalculation(j1, out_bclen);
						double val = kv._2.getValue(in_i1, in_j1);
						if(val != 0) {
							row.append(out_j1, val);
							atleastOneNonZero = true;
						}
					}
					if(atleastOneNonZero)
						blk.appendRow(out_i1, row);
				}
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(indx, blk));
			}
		}
		return retVal;
	}
	
}