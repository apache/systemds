package com.ibm.bi.dml.api.datasource.functions;

import java.io.Serializable;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

import scala.Tuple2;

import com.ibm.bi.dml.api.datasource.MLBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class GetMLBlock implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Row>, Serializable {

	private static final long serialVersionUID = 8829736765002126985L;

	@Override
	public Row call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		return new MLBlock(kv._1, kv._2);
	}
	
}