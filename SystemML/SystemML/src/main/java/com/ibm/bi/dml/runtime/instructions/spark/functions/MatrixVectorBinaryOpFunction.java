package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixVectorBinaryOpFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {
	private static final long serialVersionUID = 2814994742189295067L;
	private BinaryOperator op;
	private boolean isBroadcastRHSVar;
	private boolean isColumnVector; 
	Broadcast<PartitionedMatrixBlock> _pmV;
	boolean isOuter;
	
	public MatrixVectorBinaryOpFunction( boolean isBroadcastRHSVar, boolean isColumnVector, 
			Broadcast<PartitionedMatrixBlock> binput, BinaryOperator op, 
			int brlen, int bclen, long rdd_rlen, long rdd_clen, boolean isOuter ) {
		this.isBroadcastRHSVar = isBroadcastRHSVar;
		this.isColumnVector = isColumnVector;
		this.op = op;
		
		//get the broadcast vector
		_pmV = binput;
		this.isOuter = isOuter;
	}

	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		if(isOuter) {
			// TODO: Here we went for performance rather than robustness where assumption is we stream extremely  large 
			// moderately skinny matrices. For robustness, we can later do replication-based approach 
			// so as not to have very large result set (i.e. mapToPair instead of flatMapToPair).
			int pNumRowBlocks = _pmV.value().getNumRowBlocks();
			int pNumColsBlocks = _pmV.value().getNumColumnBlocks();
			if(pNumColsBlocks == 1) {
				for(int i = 1; i <= pNumRowBlocks; i++ ) {
					MatrixBlock [] blocks = getBlocksForBinaryOperations(kv._2, i);
					MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (op, blocks[1], new MatrixBlock()));
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(i, kv._1.getColumnIndex()), resultBlk));
				}
			}
			else {
				for(int i = 1; i <= pNumColsBlocks; i++ ) {
					MatrixBlock [] blocks = getBlocksForBinaryOperations(kv._2, i);
					MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (op, blocks[1], new MatrixBlock()));
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(kv._1.getRowIndex(), i), resultBlk));
				}
			}
		}
		else {
			MatrixBlock [] blocks = null;
			// if(kv._1.getColumnIndex() == 1)
			if(isColumnVector)
				blocks = getBlocksForBinaryOperations(kv._2, kv._1.getRowIndex());
			else
				blocks = getBlocksForBinaryOperations(kv._2, kv._1.getColumnIndex());
			
			MatrixBlock resultBlk = (MatrixBlock) (blocks[0].binaryOperations (op, blocks[1], new MatrixBlock()));
			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlk));
		}
		
		return retVal;
	}
	
	
	private MatrixBlock [] getBlocksForBinaryOperations(MatrixBlock streamingInputBlk, long index) {
		MatrixBlock blk1 = null;
		MatrixBlock blk2 = null;
		
		if(isBroadcastRHSVar) {
			blk1 = streamingInputBlk;
			if(this.isColumnVector) {
				blk2 = _pmV.value().getMatrixBlock((int) index,1);
			}
			else {
				blk2 = _pmV.value().getMatrixBlock(1, (int) index);
			}
		}
		else {
			blk2 = streamingInputBlk;
			if(this.isColumnVector) {
				blk2 = _pmV.value().getMatrixBlock((int) index,1);
			}
			else {
				blk2 = _pmV.value().getMatrixBlock(1, (int) index);
			}
		}
		
		MatrixBlock [] retVal = new MatrixBlock[2];
		retVal[0] = blk1;
		retVal[1] = blk2;
		return retVal;
	}
		
}
