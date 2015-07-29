package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;

import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class ReplicateVector implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {

	private static final long serialVersionUID = -1505557561471236851L;
	
	boolean byRow; long numReplications;
	public ReplicateVector(boolean byRow, long numReplications) {
		this.byRow = byRow;
		this.numReplications = numReplications;
	}
	
	@Override
	public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		for(int i = 1; i <= numReplications; i++) {
			if(byRow) {
				if(kv._1.getRowIndex() != 1) {
					throw new Exception("Expected a row vector in ReplicateVector");
				}
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(i, kv._1.getColumnIndex()), kv._2));
			}
			else {
				if(kv._1.getColumnIndex() != 1) {
					throw new Exception("Expected a column vector in ReplicateVector");
				}
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(kv._1.getRowIndex(), i), kv._2));
			}
		}
		return retVal;
	}

}
