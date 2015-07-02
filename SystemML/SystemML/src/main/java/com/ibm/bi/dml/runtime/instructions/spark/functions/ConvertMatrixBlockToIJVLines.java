package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.Iterator;

import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.BinaryBlockToTextCellConverter;

public class ConvertMatrixBlockToIJVLines implements FlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, String> {

	private static final long serialVersionUID = 3555147684480763957L;
	
	int brlen; int bclen;
	public ConvertMatrixBlockToIJVLines(int brlen, int bclen) {
		this.brlen = brlen;
		this.bclen = bclen;
	}
	
	@Override
	public Iterable<String> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
		final BinaryBlockToTextCellConverter converter = new BinaryBlockToTextCellConverter();
		converter.setBlockSize(brlen, bclen);
		converter.convert(kv._1, kv._2);
		
		return new Iterable<String>() {
			@Override
			public Iterator<String> iterator() {
				return new Iterator<String>() {
					
					@Override
					public void remove() {}
					
					@Override
					public String next() {
						return converter.next().getValue().toString();
					}
					
					@Override
					public boolean hasNext() {
						return converter.hasNext();
					}
				};
			}
		};
	}

}
