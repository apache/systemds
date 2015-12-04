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
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.instructions.mr.CSVWriteInstruction;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.data.TaggedFirstSecondIndexes;

public class CSVWriteMapper extends MapperBase implements Mapper<Writable, Writable, TaggedFirstSecondIndexes, MatrixBlock>
{
	
	
	HashMap<Byte, ArrayList<Byte>> inputOutputMap=new HashMap<Byte, ArrayList<Byte>>();
	TaggedFirstSecondIndexes outIndexes=new TaggedFirstSecondIndexes();
	
	@Override
	@SuppressWarnings("unchecked")
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<TaggedFirstSecondIndexes, MatrixBlock> out,
			Reporter reporter) throws IOException
	{
		long start=System.currentTimeMillis();
		
		//for each represenattive matrix, read the record and apply instructions
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			//convert the record into the right format for the representatice matrix
			inputConverter.setBlockSize(brlens[i], bclens[i]);
			inputConverter.convert(rawKey, rawValue);
			
			byte thisMatrix=representativeMatrixes.get(i);
			
			//apply unary instructions on the converted indexes and values
			while(inputConverter.hasNext())
			{
				Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
				MatrixIndexes indexes=pair.getKey();
				
				MatrixBlock value=pair.getValue();
				
				outIndexes.setIndexes(indexes.getRowIndex(), indexes.getColumnIndex());
				ArrayList<Byte> outputs=inputOutputMap.get(thisMatrix);
				for(byte output: outputs)
				{
					outIndexes.setTag(output);
					out.collect(outIndexes, value);
					//LOG.info("Mapper output: "+outIndexes+", "+value+", tag: "+output);
				}
			}
		}
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			CSVWriteInstruction[] ins = MRJobConfiguration.getCSVWriteInstructions(job);
			for(CSVWriteInstruction in: ins)
			{
				ArrayList<Byte> outputs=inputOutputMap.get(in.input);
				if(outputs==null)
				{
					outputs=new ArrayList<Byte>();
					inputOutputMap.put(in.input, outputs);
				}
				outputs.add(in.output);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		// do nothing
	}
	
}
