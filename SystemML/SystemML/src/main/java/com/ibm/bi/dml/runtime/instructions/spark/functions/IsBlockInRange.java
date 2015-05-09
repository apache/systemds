package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

public class IsBlockInRange implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> {
	private static final long serialVersionUID = 5849687296021280540L;
	
	long rl; long ru; long cl; long cu;
	int brlen; int bclen;
	
	public IsBlockInRange(long rl, long ru, long cl, long cu, int brlen, int bclen) {
		this.rl = rl;
		this.ru = ru;
		this.cl = cl;
		this.cu = cu;
		this.brlen = brlen;
		this.bclen = bclen;
	}

	@Override
	public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		long brIndex = kv._1.getRowIndex();
		long bcIndex = kv._1.getColumnIndex();
	
		long bRLowerIndex = (brIndex-1)*brlen + 1;
		long bRUpperIndex = brIndex*brlen;
		long bCLowerIndex = (bcIndex-1)*bclen + 1;
		long bCUpperIndex = bcIndex*bclen;
		
		if(rl > bRUpperIndex || ru < bRLowerIndex) {
			return false;
		}
		else if(cl > bCUpperIndex || cu < bCLowerIndex) {
			return false;
		}
		else {
			return true;
		}
	}
}
