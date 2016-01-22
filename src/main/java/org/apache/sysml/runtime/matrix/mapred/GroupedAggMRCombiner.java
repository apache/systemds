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
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.matrix.data.TaggedMatrixIndexes;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class GroupedAggMRCombiner extends ReduceBase
	implements Reducer<TaggedMatrixIndexes, WeightedCell, TaggedMatrixIndexes, WeightedCell>
{	
	//grouped aggregate instructions
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInstructions = new HashMap<Byte, GroupedAggregateInstruction>();
	
	//reused intermediate objects
	private CM_COV_Object cmObj = new CM_COV_Object(); 
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private WeightedCell outCell = new WeightedCell();

	@Override
	public void reduce(TaggedMatrixIndexes key, Iterator<WeightedCell> values,
			           OutputCollector<TaggedMatrixIndexes, WeightedCell> out, Reporter reporter)
		throws IOException 
	{
		long start = System.currentTimeMillis();
		
		//get aggregate operator
		GroupedAggregateInstruction ins = grpaggInstructions.get(key.getTag());
		Operator op = ins.getOperator();
		boolean isPartialAgg = true;
		
		//combine iterator to single value
		try
		{
			if(op instanceof CMOperator) //everything except sum
			{
				if( ((CMOperator) op).isPartialAggregateOperator() )
				{
					cmObj.reset();
					CM lcmFn = cmFn.get(key.getTag());
					
					//partial aggregate cm operator 
					while( values.hasNext() )
					{
						WeightedCell value=values.next();
						lcmFn.execute(cmObj, value.getValue(), value.getWeight());				
					}
					
					outCell.setValue(cmObj.getRequiredPartialResult(op));
					outCell.setWeight(cmObj.getWeight());	
				}
				else //forward tuples to reducer
				{
					isPartialAgg = false; 
					while( values.hasNext() )
						out.collect(key, values.next());
				}				
			}
			else if(op instanceof AggregateOperator) //sum
			{
				AggregateOperator aggop=(AggregateOperator) op;
					
				if( aggop.correctionExists )
				{
					KahanObject buffer=new KahanObject(aggop.initialValue, 0);
					
					KahanPlus.getKahanPlusFnObject();
					
					//partial aggregate with correction
					while( values.hasNext() )
					{
						WeightedCell value=values.next();
						aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
					}
					
					outCell.setValue(buffer._sum);
					outCell.setWeight(1);
				}
				else //no correction
				{
					double v = aggop.initialValue;
					
					//partial aggregate without correction
					while(values.hasNext())
					{
						WeightedCell value=values.next();
						v=aggop.increOp.fn.execute(v, value.getValue()*value.getWeight());
					}
					
					outCell.setValue(v);
					outCell.setWeight(1);
				}				
			}
			else
				throw new IOException("Unsupported operator in instruction: " + ins);
		}
		catch(Exception ex)
		{
			throw new IOException(ex);
		}
		
		//collect the output (to reducer)
		if( isPartialAgg )
			out.collect(key, outCell);
		
		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}

	@Override
	public void configure(JobConf job)
	{
		try 
		{
			GroupedAggregateInstruction[] grpaggIns = MRJobConfiguration.getGroupedAggregateInstructions(job);
			if( grpaggIns != null )	
				for(GroupedAggregateInstruction ins : grpaggIns)
				{
					grpaggInstructions.put(ins.output, ins);	
					if( ins.getOperator() instanceof CMOperator )
						cmFn.put(ins.output, CM.getCMFnObject(((CMOperator)ins.getOperator()).getAggOpType()));
				}
		} 
		catch (Exception e) 
		{
			throw new RuntimeException(e);
		} 
	}
	
	@Override
	public void close()
	{
		//do nothing, overrides unnecessary handling in superclass
	}
}
