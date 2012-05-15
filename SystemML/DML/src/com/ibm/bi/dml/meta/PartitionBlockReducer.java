package com.ibm.bi.dml.meta;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;


public class PartitionBlockReducer extends MapReduceBase 
implements Reducer<IntWritable, MatrixBlock, MatrixIndexes, MatrixValue>{

	MatrixBlock block = new MatrixBlock() ;
	MatrixIndexes indexes = new MatrixIndexes() ;
	protected MultipleOutputs multipleOutputs;
	long counter = 1 ;
	
	@Override
	public void reduce(IntWritable pair, Iterator<MatrixBlock> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter)
	throws IOException {
		counter = 1;
		while(values.hasNext()) {
			block = values.next();
			indexes.setIndexes(counter, 1) ;	//systemml matrxblks start from (1,1)
			counter++ ;
			reporter.incrCounter("counter", "" + pair.get(), 1) ;
			multipleOutputs.getCollector("" + pair.get(), reporter).collect(indexes, block) ;
		}
	}

	public void close() throws IOException {
		multipleOutputs.close();
	}

	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job) ;
	}
}