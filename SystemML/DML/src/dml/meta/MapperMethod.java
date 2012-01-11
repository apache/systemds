package dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;

import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.PartialBlock;

public abstract class MapperMethod {
	LongWritable pols = new LongWritable() ;
	PartialBlock partialBuffer = new PartialBlock() ;
	PartitionParams pp ;
	
	public MapperMethod(PartitionParams pp) {
		this.pp = pp ;
	}
	
	abstract void execute(Pair<MatrixIndexes, MatrixValue> pair, OutputCollector out) 
	throws IOException ;
}
