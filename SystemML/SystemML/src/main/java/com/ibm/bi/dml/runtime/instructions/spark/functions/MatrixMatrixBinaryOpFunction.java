package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

public class MatrixMatrixBinaryOpFunction implements Function<Tuple2<MatrixBlock,MatrixBlock>, MatrixBlock> {
	private static final long serialVersionUID = -2683276102742977900L;
	
	private BinaryOperator op;
	
	public MatrixMatrixBinaryOpFunction(BinaryOperator op) {
		this.op = op;
	}

	@Override
	public MatrixBlock call(Tuple2<MatrixBlock, MatrixBlock> kv)
			throws Exception {
		MatrixBlock blk1 = kv._1;
		MatrixBlock blk2 = kv._2;
		MatrixBlock resultBlk = (MatrixBlock) (blk1.binaryOperations (op, blk2, new MatrixBlock()));
		return resultBlk;
	}
	
}