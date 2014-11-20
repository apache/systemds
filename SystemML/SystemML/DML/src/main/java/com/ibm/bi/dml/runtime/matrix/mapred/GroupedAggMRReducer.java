/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.data.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class GroupedAggMRReducer extends ReduceBase
	implements Reducer<TaggedInt, WeightedCell, MatrixIndexes, MatrixCell >
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInstructions=new HashMap<Byte, GroupedAggregateInstruction>();
	private CM_COV_Object cmObj=new CM_COV_Object(); 
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private HashMap<Byte, Vector<Integer>> outputIndexesMapping=new HashMap<Byte, Vector<Integer>>();
	
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
