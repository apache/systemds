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

public class ReconstructionHashMapReducer extends MapReduceBase 
implements Reducer<LongWritable, ReconstructionHashMapMapOutputValue, MatrixIndexes, MatrixValue>{
	protected MultipleOutputs multipleOutputs;
	PartitionParams pp = new PartitionParams() ;
	
	@Override
	public void reduce(LongWritable key, Iterator<ReconstructionHashMapMapOutputValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter)
	throws IOException {
		//reconstruct the matrix block from the subrows/subcols
		MatrixIndexes indexes = new MatrixIndexes(key.get() + 1, 1);		//systemml matrixblks start from (1,1)
		MatrixBlock thisblock = new MatrixBlock(pp.rows_in_block, pp.columns_in_block, true); //presume sparse
		thisblock.setMaxColumn(1);	//single col matrix	//do we need this?
		int numrows = 0;
		//TODO: recheck about maxrow vs rlen!
		while(values.hasNext()) {
			ReconstructionHashMapMapOutputValue tmp = new ReconstructionHashMapMapOutputValue(values.next());
			thisblock.setValue(tmp.subrowid, 0, tmp.entry);
			reporter.incrCounter("counter", "", 1) ;	//incrmt num subrowblks for use in driver! only 1 ctr since 1 o/p!
			numrows++;
		}
		thisblock.setMaxRow(numrows);	//do we need this? TODO
		//thisblock.examSparsity();	//refactor based on actual sparsity
		System.out.println("$$$$$$$ reconstructed using hashmap, indexes: " + indexes.toString());
		multipleOutputs.getCollector("", reporter).collect(indexes, thisblock) ;
	}

	public void close() throws IOException {
		multipleOutputs.close();
	}
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job);
		pp = MRJobConfiguration.getPartitionParams(job);
	}
}