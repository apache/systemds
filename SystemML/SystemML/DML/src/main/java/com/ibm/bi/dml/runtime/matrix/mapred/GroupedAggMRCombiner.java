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

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.data.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.data.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class GroupedAggMRCombiner extends ReduceBase
	implements Reducer<TaggedInt, WeightedCell, TaggedInt, WeightedCell>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//grouped aggregate instructions
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInstructions = new HashMap<Byte, GroupedAggregateInstruction>();
	
	//reused intermediate objects
	private CM_COV_Object cmObj = new CM_COV_Object(); 
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private WeightedCell outCell = new WeightedCell();

	@Override
	public void reduce(TaggedInt key, Iterator<WeightedCell> values,
			           OutputCollector<TaggedInt, WeightedCell> out, Reporter reporter)
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
