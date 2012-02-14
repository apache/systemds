package dml.meta;

import java.io.IOException;

import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;

public abstract class BlockMapperMethod {
	MatrixIndexes mi = new MatrixIndexes() ;
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;
	
	public BlockMapperMethod(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
	}
	
	abstract void execute(Well1024a currRandom, Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out)
	throws IOException ;
}
