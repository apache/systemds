package com.ibm.bi.dml.runtime.matrix.mapred;

import java.util.HashMap;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class MRBaseForCommonInstructions extends MapReduceBase{

	//indicate whether the matrix value in this mapper is a matrix cell or a matrix block
	protected Class<? extends MatrixValue> valueClass;
	
	//a cache to hold the intermediate results
	protected CachedValueMap cachedValues=new CachedValueMap();
	
	protected HashMap<Byte, MatrixCharacteristics> dimensions=new HashMap<Byte, MatrixCharacteristics>();
	
	//temporary variables
	protected IndexedMatrixValue tempValue=null;
	protected IndexedMatrixValue zeroInput=null;

	public void configure(JobConf job)
	{	
		//whether to use the cell representation or the block representation
		valueClass=MRJobConfiguration.getMatrixValueClass(job);
		//allocate space for temporary variables
		tempValue=new IndexedMatrixValue(valueClass);
		zeroInput=new IndexedMatrixValue(valueClass);
		
		byte[] diagm2vIndexes=MRJobConfiguration.getIntermediateMatrixIndexes(job);
		if(diagm2vIndexes!=null)
		{
			for(byte index: diagm2vIndexes)
				dimensions.put(index, MRJobConfiguration.getIntermediateMatrixCharactristics(job, index));
		}
	}

	//process mixture of instructions
	protected void processMixedInstructions(MRInstruction[] mixed_instructions) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(mixed_instructions==null)
			return;
		for(MRInstruction ins: mixed_instructions)
			processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
	}
	
	protected void processMixedInstructions(Vector<MRInstruction> mixed_instructions) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(mixed_instructions==null || mixed_instructions.isEmpty())
			return;
		for(MRInstruction ins: mixed_instructions)
			processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
	}
	
	protected void processOneInstruction(MRInstruction ins, Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if(ins instanceof ZeroOutInstruction || ins instanceof AggregateUnaryInstruction || ins instanceof RangeBasedReIndexInstruction)
		{
			byte input=((UnaryMRInstructionBase) ins).input;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}else
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, -1, -1);
	}
}
