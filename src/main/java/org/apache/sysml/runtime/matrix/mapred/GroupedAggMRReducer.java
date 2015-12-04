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
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.TaggedInt;
import org.apache.sysml.runtime.matrix.data.WeightedCell;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class GroupedAggMRReducer extends ReduceBase
	implements Reducer<TaggedInt, WeightedCell, MatrixIndexes, MatrixCell >
{
	
	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInstructions=new HashMap<Byte, GroupedAggregateInstruction>();
	private CM_COV_Object cmObj=new CM_COV_Object(); 
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private HashMap<Byte, ArrayList<Integer>> outputIndexesMapping=new HashMap<Byte, ArrayList<Integer>>();
	
	@Override
	public void reduce(TaggedInt key,Iterator<WeightedCell> values,
			OutputCollector<MatrixIndexes, MatrixCell> out, Reporter report)
		throws IOException 
	{
		commonSetup(report);
		
		//get operator
		GroupedAggregateInstruction ins = grpaggInstructions.get(key.getTag());
		Operator op = ins.getOperator();
		
		try
		{
			if(op instanceof CMOperator) //all, but sum
			{
				cmObj.reset();
				CM lcmFn = cmFn.get(key.getTag());
				while(values.hasNext())
				{
					WeightedCell value=values.next();
					lcmFn.execute(cmObj, value.getValue(), value.getWeight());
				}
				outCell.setValue(cmObj.getRequiredResult(op));				
			}
			else if(op instanceof AggregateOperator) //sum
			{
				AggregateOperator aggop=(AggregateOperator) op;
					
				if(aggop.correctionExists)
				{
					KahanObject buffer=new KahanObject(aggop.initialValue, 0);
					while(values.hasNext())
					{
						WeightedCell value=values.next();
						aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
					}
					outCell.setValue(buffer._sum);
				}
				else
				{
					double v=aggop.initialValue;
					while(values.hasNext())
					{
						WeightedCell value=values.next();
						v=aggop.increOp.fn.execute(v, value.getValue()*value.getWeight());
					}
					outCell.setValue(v);
				}
				
			}
			else
				throw new IOException("Unsupported operator in instruction: " + ins);
		}
		catch(Exception ex)
		{
			throw new IOException(ex);
		}
		
		outIndex.setIndexes((long)key.getBaseObject().get(), 1);
		cachedValues.reset();
		cachedValues.set(key.getTag(), outIndex, outCell);
		processReducerInstructions();
		
		//output the final result matrices
		outputResultsFromCachedValues(report);
	}

	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		
		try 
		{
			GroupedAggregateInstruction[] grpaggIns=MRJobConfiguration.getGroupedAggregateInstructions(job);
			if(grpaggIns==null)
				throw new RuntimeException("no GroupAggregate Instructions found!");
			
			for(GroupedAggregateInstruction ins: grpaggIns)
			{
				grpaggInstructions.put(ins.output, ins);	
				if( ins.getOperator() instanceof CMOperator )
					cmFn.put(ins.output, CM.getCMFnObject(((CMOperator)ins.getOperator()).getAggOpType()));
				outputIndexesMapping.put(ins.output, getOutputIndexes(ins.output));
			}
		} 
		catch (Exception e) 
		{
			throw new RuntimeException(e);
		} 
	}
	
	@Override
	public void close() 
		throws IOException
	{
		super.close();
	}
}
