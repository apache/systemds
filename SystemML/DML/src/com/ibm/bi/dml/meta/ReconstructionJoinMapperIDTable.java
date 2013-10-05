/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class ReconstructionJoinMapperIDTable extends MapReduceBase
implements Mapper<Writable, Writable, LongWritable, ReconstructionJoinMapOutputValue> 
{ //TODO change read key/val format

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//private Converter inputConverter=null;	//do i need this for mine?? to get seq file key/val format correctly?!
	PartitionParams pp = new PartitionParams() ;
	MultipleOutputs multipleOutputs ;
	int thisfold;

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<LongWritable, ReconstructionJoinMapOutputValue> out, Reporter reporter)
	throws IOException {
		WritableLongArray tuple = new WritableLongArray((WritableLongArray)rawValue);
		LongWritable tuplekey = (LongWritable)rawKey;
		ReconstructionJoinMapOutputValue outval = new ReconstructionJoinMapOutputValue();
		//TODO: note that the reconstruction assumes it is cv holdout on col or el rsm
		if(tuple.array[thisfold] < 0) {	//only train data considered
			LongWritable outkey = new LongWritable(-1*tuple.array[thisfold] - 1);	//due to encoding of futrowid
			outval.rowid = tuplekey.get();	//the orig colid is value; if this is -1, then incoming is matrx elem
			out.collect(outkey, outval);	//so mk keyval pairs are snet out, rather than just m before!
		}		
	}
	
	public void close() throws IOException  {
		multipleOutputs.close();
	}
	//TODO: change the reading configs and format for seq file reading in!!!
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job) ;
		pp = MRJobConfiguration.getPartitionParams(job) ;
		thisfold = job.getInt("foldnum", 0);	//get the fold num corresp to this train o/p col matrx		
	}
}
