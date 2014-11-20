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


public class ReconstructionHashMapReducer extends MapReduceBase 
implements Reducer<LongWritable, ReconstructionHashMapMapOutputValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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