package dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import umontreal.iro.lecuyer.rng.WELL1024;

public abstract class ReducerMethod {
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;
	MatrixIndexes mi = new MatrixIndexes();
	
	public ReducerMethod(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
	}
	
	abstract void execute(WELL1024 currRandom, LongWritable pair, MatrixBlock block, Reporter reporter) throws IOException ;
}
