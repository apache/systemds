/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CM_COV_Object;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.TaggedInt;
import com.ibm.bi.dml.runtime.matrix.io.WeightedCell;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public class GroupedAggMRReducer extends ReduceBase
implements Reducer<TaggedInt, WeightedCell, MatrixIndexes, MatrixCell >
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixIndexes outIndex=new MatrixIndexes(1, 1);
	private MatrixCell outCell=new MatrixCell();
	private HashMap<Byte, GroupedAggregateInstruction> grpaggInstructions=new HashMap<Byte, GroupedAggregateInstruction>();
	private CM_COV_Object cmObj=new CM_COV_Object(); 
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private HashMap<Byte, Vector<Integer>> outputIndexesMapping=new HashMap<Byte, Vector<Integer>>();
	@Override
	public void reduce(TaggedInt key,
			Iterator<WeightedCell> values,
			OutputCollector<MatrixIndexes, MatrixCell> out, Reporter report)
			throws IOException {
		commonSetup(report);
		GroupedAggregateInstruction ins=grpaggInstructions.get(key.getTag());
		Operator op=ins.getOperator();
		if(op instanceof CMOperator)
		{
			cmObj.reset();
			CM lcmFn = cmFn.get(key.getTag());
			while(values.hasNext())
			{
				WeightedCell value=values.next();
				try {
					lcmFn.execute(cmObj, value.getValue(), value.getWeight());
				} catch (DMLRuntimeException e) {
					throw new IOException(e);
				}
			}
			try {
				outCell.setValue(cmObj.getRequiredResult(op));
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
			
		}else if(op instanceof AggregateOperator)
		{
			AggregateOperator aggop=(AggregateOperator) op;
				
			if(aggop.correctionExists)
			{
				KahanObject buffer=new KahanObject(aggop.initialValue, 0);
				while(values.hasNext())
				{
					WeightedCell value=values.next();
					try {
						aggop.increOp.fn.execute(buffer, value.getValue()*value.getWeight());
					} catch (DMLRuntimeException e) {
						throw new IOException(e);
					}
				}
				outCell.setValue(buffer._sum);
			}
			else
			{
				double v=aggop.initialValue;
				while(values.hasNext())
				{
					WeightedCell value=values.next();
					try {
						v=aggop.increOp.fn.execute(v, value.getValue()*value.getWeight());
					} catch (DMLRuntimeException e) {
						throw new IOException(e);
					}
				}
				outCell.setValue(v);
			}
			
		}else
			throw new IOException("cannot execute instruciton "+ins);
		
		outIndex.setIndexes((long)key.getBaseObject().get(), 1);
		cachedValues.reset();
		cachedValues.set(key.getTag(), outIndex, outCell);
		//System.out.println("after cm: "+outIndex+" -- "+outCell);
		processReducerInstructions();
		//output the final result matrices
		outputResultsFromCachedValues(report);
		
	/*	Vector<Integer> outputIndexes = outputIndexesMapping.get(key.getTag());
		for(int i: outputIndexes)
		{
			collectOutput_N_Increase_Counter(outIndex, outCell, i, report);
			//System.out.println("final output: "+outIndex+" -- "+outCell);
		}*/
		
	}

	public void configure(JobConf job)
	{
		super.configure(job);
		//valueClass=MatrixCell.class;
		try {
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
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
	}
}
