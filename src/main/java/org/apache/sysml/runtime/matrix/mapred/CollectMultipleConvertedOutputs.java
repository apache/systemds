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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.Pair;


@SuppressWarnings("rawtypes")
public class CollectMultipleConvertedOutputs 
{
	protected Converter[] outputConverters;
	protected MultipleOutputs multipleOutputs;
	protected MatrixCharacteristics[] matrixStats;
	
	public CollectMultipleConvertedOutputs(Converter[] convts, MatrixCharacteristics[] stats, 
			MultipleOutputs outputs)
	{
		outputConverters=convts;
		multipleOutputs=outputs;
		matrixStats=stats;
	}
	
	@SuppressWarnings("unchecked")
	public void collectOutput(Writable key, Writable value, int output, Reporter reporter) 
	throws IOException
	{
		Converter<Writable, Writable, Writable, Writable> conv=outputConverters[output];
		conv.setBlockSize(matrixStats[output].getRowsPerBlock(), matrixStats[output].getColsPerBlock());
		conv.convert(key, value);
	//	System.out.println("output before convert: "+key+" "+value);
		while(conv.hasNext())
		{
			Pair<Writable, Writable> outpair=conv.next();
			multipleOutputs.getCollector(Integer.toString(output), reporter).collect(outpair.getKey(), outpair.getValue());
		//	System.out.println("output in collectOutput "+outpair.getKey().toString()+":"+outpair.getValue());
		}
	}
	
	@SuppressWarnings("unchecked")
	public void directOutput(Writable key, Writable value, int output, Reporter reporter) 
		throws IOException
	{
		multipleOutputs.getCollector(Integer.toString(output), reporter).collect(key, value);
	}

	public void close() 
		throws IOException 
	{	
		multipleOutputs.close();
	}

}
