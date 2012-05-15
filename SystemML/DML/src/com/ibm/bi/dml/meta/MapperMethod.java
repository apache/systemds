package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.PartialBlock;


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
