package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.functionobjects.CM;
import com.ibm.bi.dml.runtime.functionobjects.COV;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.matrix.io.CM_N_COVCell;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.io.WeightedPair;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.COVOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class CMCOVMRMapper extends MapperBase
implements Mapper<Writable, Writable, Writable, Writable>{

	private boolean firsttime=true;
	private HashMap<Byte, CM> cmFn = new HashMap<Byte, CM>();
	private COV covFn=COV.getCOMFnObject();
	private OutputCollector<Writable, Writable> cachedCollector=null;
	private CachedValueMap cmNcovCache=new CachedValueMap();
	protected HashSet<Byte> cmTags=new HashSet<Byte>();
	protected HashSet<Byte> covTags=new HashSet<Byte>();
	@Override
	public void map(Writable index, Writable cell,
			OutputCollector<Writable, Writable> out, Reporter report)
			throws IOException {
		if(firsttime)
		{
			cachedCollector=out;
			firsttime=false;
		}
	//	System.out.println("input: "+index+" -- "+cell);
		commonMap(index, cell, out, report);
	}

	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
			throws IOException {
		
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		for(byte tag: cmTags)
		{
			CM lcmFn = cmFn.get(tag);
			IndexedMatrixValue input = cachedValues.getFirst(tag);
			if(input==null)
				continue;
			WeightedPair inputPair=(WeightedPair)input.getValue();
			CM_N_COVCell cmValue = (CM_N_COVCell) cmNcovCache.getFirst(tag).getValue();
			try {
				
			//	System.out.println("~~~~~\nold: "+cmValue.getCM_N_COVObject());
			//	System.out.println("add: "+inputPair);
				lcmFn.execute(cmValue.getCM_N_COVObject(), inputPair.getValue(), inputPair.getWeight());
			//	System.out.println("new: "+cmValue.getCM_N_COVObject());
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}
		
		for(byte tag: covTags)
		{
			IndexedMatrixValue input = cachedValues.getFirst(tag);
			if(input==null)
				continue;
			//System.out.println("*** cached Value:\n"+cachedValues);
			WeightedPair inputPair=(WeightedPair)input.getValue();
			CM_N_COVCell comValue = (CM_N_COVCell) cmNcovCache.getFirst(tag).getValue();
			try {
				
				//System.out.println("~~~~~\nold: "+comValue.getCM_N_COVObject());
			//	System.out.println("add: "+inputPair);
				covFn.execute(comValue.getCM_N_COVObject(), inputPair.getValue(),  inputPair.getOtherValue(), inputPair.getWeight());
			//	System.out.println("new: "+comValue.getCM_N_COVObject());
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
		}
	}
	
	public void close() throws IOException
	{
		if(cachedCollector!=null)
		{
			for(byte tag: cmTags)
			{
				CM_N_COVCell cmValue = (CM_N_COVCell) cmNcovCache.getFirst(tag).getValue();
				cachedCollector.collect(new TaggedFirstSecondIndexes(1, tag, 1), cmValue);
				//System.out.println("output to reducer with tag:"+tag+" and value: "+cmValue);
			}
			
			for(byte tag: covTags)
			{
				CM_N_COVCell comValue = (CM_N_COVCell) cmNcovCache.getFirst(tag).getValue();
				cachedCollector.collect(new TaggedFirstSecondIndexes(1, tag, 1), comValue);
				//System.out.println("output to reducer with tag:"+tag+" and value: "+comValue);
			}
		}
	}

	public void configure(JobConf job)
	{
		super.configure(job);
		try {
			CM_N_COVInstruction[] cmIns=MRJobConfiguration.getCM_N_COVInstructions(job);
			for(CM_N_COVInstruction ins: cmIns)
			{
				if(ins.getOperator() instanceof COVOperator)
					covTags.add(ins.input);
				else //CMOperator
				{
					cmTags.add(ins.input);
					cmFn.put(ins.input, CM.getCMFnObject(((CMOperator)ins.getOperator()).getAggOpType()));
				}
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		for(byte tag: cmTags)
		{
			cmNcovCache.holdPlace(tag, CM_N_COVCell.class);
		}
		
		for(byte tag: covTags)
		{
			cmNcovCache.holdPlace(tag, CM_N_COVCell.class);
		}
	}
}
