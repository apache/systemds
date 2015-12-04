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

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.CombineUnaryInstruction;
import org.apache.sysml.runtime.matrix.SortMR;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;

@SuppressWarnings("rawtypes")
public class ValueSortMapper<KIN extends WritableComparable, VIN extends Writable, KOUT extends WritableComparable, VOUT extends Writable> extends MapReduceBase 
      implements Mapper<KIN, VIN, KOUT, VOUT>
{
	
	private int brlen;
	private int bclen;
	private CombineUnaryInstruction combineInstruction=null;
	private Converter<KIN, VIN, KOUT, VOUT> inputConverter;
	private IntWritable one=new IntWritable(1);
	private DoubleWritable combinedKey=new DoubleWritable();
	
	@SuppressWarnings("unchecked")
	public void map(KIN key, VIN value, OutputCollector<KOUT, VOUT> out,
			Reporter reporter) throws IOException {
		inputConverter.convert(key, value);
		while(inputConverter.hasNext())
		{
			Pair pair=inputConverter.next();
			if(combineInstruction==null)
			{
				//System.out.println("output: "+pair.getKey()+": "+pair.getValue());
				out.collect((KOUT) pair.getKey(), (VOUT)pair.getValue());
			}else
			{
				processCombineUnaryInstruction(pair, out);
			}
		}
	} 
	
	@SuppressWarnings("unchecked")
	private void processCombineUnaryInstruction(Pair pair, OutputCollector<KOUT, VOUT> out) 
		throws IOException
	{
		combinedKey.set(((MatrixCell)pair.getValue()).getValue());
		out.collect((KOUT)combinedKey, (VOUT)one);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public void configure(JobConf job)
	{
		try 
		{
			brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
			bclen = MRJobConfiguration.getNumColumnsPerBlock(job, (byte) 0);
			String str=job.get(SortMR.COMBINE_INSTRUCTION, null);
			if(str!=null && !str.isEmpty() && !"null".equals(str))
					combineInstruction=(CombineUnaryInstruction) CombineUnaryInstruction.parseInstruction(str);
			inputConverter = MRJobConfiguration.getInputConverter(job, (byte) 0);
			inputConverter.setBlockSize(brlen, bclen);
		} 
		catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}
}
