/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;


public class MMCJMRCombiner extends MMCJMRCombinerReducerBase 
implements Reducer<TaggedFirstSecondIndexes, MatrixValue, TaggedFirstSecondIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public void reduce(TaggedFirstSecondIndexes indexes, Iterator<MatrixValue> values,
			OutputCollector<TaggedFirstSecondIndexes, MatrixValue> out,
			Reporter report) throws IOException {
	
		//perform aggregate
		MatrixValue aggregateValue=performAggregateInstructions(indexes, values);
		
		if(aggregateValue!=null)
			out.collect(indexes, aggregateValue);
		
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
	}
}
