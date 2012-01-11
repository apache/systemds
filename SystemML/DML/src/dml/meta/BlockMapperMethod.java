package dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.PartialBlock;

public abstract class BlockMapperMethod {
	MatrixIndexes mi = new MatrixIndexes() ;
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;
	
	public BlockMapperMethod(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
	}
	
	abstract void execute(WELL1024 currRandom, Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out)
	throws IOException ;
}
