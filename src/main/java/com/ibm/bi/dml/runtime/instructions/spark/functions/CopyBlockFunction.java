package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class CopyBlockFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixBlock>,MatrixIndexes, MatrixBlock> 
{
	private static final long serialVersionUID = -196553327495233360L;

	public CopyBlockFunction(  )
	{
	
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> call(
		Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
		
		MatrixIndexes ix = new MatrixIndexes(arg0._1());
		MatrixBlock block = new MatrixBlock();
		block.copy(arg0._2());
		return new Tuple2<MatrixIndexes,MatrixBlock>(ix,block);
	}
}