/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

@SuppressWarnings("rawtypes")
public class ValueSortReducer<K extends WritableComparable, V extends Writable> extends MapReduceBase 
      implements Reducer<K, V, K, V>
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String taskID=null;
	private boolean valueIsWeight=false;
	private long count=0;
	
	public void configure(JobConf job)
	{
		taskID=MapReduceTool.getUniqueKeyPerTask(job, false);
		valueIsWeight=job.getBoolean(SortMR.VALUE_IS_WEIGHT, false);
	}

	@Override
	public void reduce(K key, Iterator<V> values, OutputCollector<K, V> out,
			Reporter report) throws IOException {
		int sum=0;
		while(values.hasNext())
		{
			V value=values.next();
			out.collect(key, value);
			if(valueIsWeight)
				sum+=((IntWritable)value).get();
			else
				sum++;
		}
		count+=sum;
		report.incrCounter(SortMR.NUM_VALUES_PREFIX, taskID, sum);
	}
}
