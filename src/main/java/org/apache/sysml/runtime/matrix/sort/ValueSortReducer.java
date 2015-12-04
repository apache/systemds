/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.matrix.sort;

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

import org.apache.sysml.runtime.matrix.SortMR;
import org.apache.sysml.runtime.util.MapReduceTool;

@SuppressWarnings("rawtypes")
public class ValueSortReducer<K extends WritableComparable, V extends Writable> extends MapReduceBase 
      implements Reducer<K, V, K, V>
{	
	
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
