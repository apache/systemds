package dml.meta;

import java.io.File;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

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

public class ReconstructionJoinReducer extends MapReduceBase 
implements Reducer<LongWritable, ReconstructionJoinMapOutputValue, MatrixIndexes, MatrixValue>{
	protected MultipleOutputs multipleOutputs;
	PartitionParams pp = new PartitionParams() ;
	
	@Override
	public void reduce(LongWritable key, Iterator<ReconstructionJoinMapOutputValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter)
	throws IOException {
		//effect the join between the matrix dbl entry and origrowid for this futrowid key
		long rowid = 0;
		double entry = 0;
		while(values.hasNext()) {	//the iterator shld have only two values!
			ReconstructionJoinMapOutputValue val = new ReconstructionJoinMapOutputValue(values.next());
			if(val.rowid == -1)		//matrix element
				entry = val.entry;
			else	//origrowid
				rowid = val.rowid;
		}
		MatrixIndexes indexes = new MatrixIndexes(rowid + 1, 1);	//single col matrx; 	//systemml matrixblks start from (1,1)
		MatrixBlock outblk = new MatrixBlock(1, 1, false);	//1x1 matrix blk
		outblk.setValue(0, 0, entry);
		reporter.incrCounter("counter", "", 1) ;
		multipleOutputs.getCollector("" , reporter).collect(indexes, outblk) ;
	}
	//TODO: we need to insert a reblock after this step at a higher level (subrowblk to blk)
	//TODO: after the reducer, check the mrrunjob for optimizations!!
	

	public void close() throws IOException {
		multipleOutputs.close();
	}
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job);
		pp = MRJobConfiguration.getPartitionParams(job);
	}
}