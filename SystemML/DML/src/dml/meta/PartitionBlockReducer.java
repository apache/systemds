package dml.meta;

import java.io.File;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;

import dml.runtime.matrix.io.MatrixBlock1D;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.MatrixValue ;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.PartialBlock;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.matrix.mapred.ReblockReducer;
import dml.runtime.util.MapReduceTool;

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