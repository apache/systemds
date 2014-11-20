/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class ReconstructionJoinReducer extends MapReduceBase 
implements Reducer<LongWritable, ReconstructionJoinMapOutputValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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