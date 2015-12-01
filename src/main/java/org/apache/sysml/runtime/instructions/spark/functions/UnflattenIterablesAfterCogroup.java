package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class UnflattenIterablesAfterCogroup implements PairFunction<Tuple2<MatrixIndexes,Tuple2<Iterable<MatrixBlock>,Iterable<MatrixBlock>>>, MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> {

	private static final long serialVersionUID = 5367350062892272775L;

	@Override
	public Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> call(
			Tuple2<MatrixIndexes, Tuple2<Iterable<MatrixBlock>, Iterable<MatrixBlock>>> arg)
			throws Exception {
		MatrixBlock left = null;
		MatrixBlock right = null;
		for(MatrixBlock blk : arg._2._1) {
			if(left == null) {
				left = blk;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		for(MatrixBlock blk : arg._2._2) {
			if(right == null) {
				right = blk;
			}
			else {
				throw new Exception("More than 1 block with same MatrixIndexes");
			}
		}
		return new Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>(arg._1, new Tuple2<MatrixBlock, MatrixBlock>(left, right));
	}
	
}