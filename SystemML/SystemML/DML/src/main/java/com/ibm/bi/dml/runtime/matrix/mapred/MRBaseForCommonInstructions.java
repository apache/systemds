/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Vector;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendMInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendGInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryMInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CumsumAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CumsumSplitInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MatrixReshapeMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RemoveEmptyMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;


public class MRBaseForCommonInstructions extends MapReduceBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//indicate whether the matrix value in this mapper is a matrix cell or a matrix block
	protected Class<? extends MatrixValue> valueClass;
	
	//a cache to hold the intermediate results
	protected CachedValueMap cachedValues=new CachedValueMap();
	
	//distributed cache data handling
	public static boolean isJobLocal = false; //set from MapperBase
	public static HashMap<Byte, DistributedCacheInput> dcValues = new HashMap<Byte, DistributedCacheInput>();
 	
	protected HashMap<Byte, MatrixCharacteristics> dimensions=new HashMap<Byte, MatrixCharacteristics>();
	
	//temporary variables
	protected IndexedMatrixValue tempValue=null;
	protected IndexedMatrixValue zeroInput=null;	

	@Override
	public void configure(JobConf job)
	{	
		//whether to use the cell representation or the block representation
		valueClass=MRJobConfiguration.getMatrixValueClass(job);
		//allocate space for temporary variables
		tempValue=new IndexedMatrixValue(valueClass);
		zeroInput=new IndexedMatrixValue(valueClass);
		
		//matrix characteristics inputs/outputs
		byte[] inputIX = MRJobConfiguration.getInputIndexesInMapper(job);
		for( byte ix : inputIX )
			dimensions.put(ix, MRJobConfiguration.getMatrixCharacteristicsForInput(job, ix));	
		byte[] outputIX = MRJobConfiguration.getOutputIndexesInMapper(job);
		for( byte ix : outputIX )
			dimensions.put(ix, MRJobConfiguration.getMatrixCharacteristicsForOutput(job, ix));	
		
		//matrix characteristics intermediates
		byte[] immediateIndexes=MRJobConfiguration.getIntermediateMatrixIndexes(job);
		if(immediateIndexes!=null)
		{
			for(byte index: immediateIndexes)
				dimensions.put(index, MRJobConfiguration.getIntermediateMatrixCharactristics(job, index));			
		}
	}
	
	/**
	 * 
	 * @param indexes
	 * @param value
	 * @param i
	 * @param reporter
	 * @param collectFinalMultipleOutputs
	 * @param resultDimsUnknown
	 * @param resultsNonZeros
	 * @param resultsMaxRowDims
	 * @param resultsMaxColDims
	 * @throws IOException
	 */
	protected void collectOutput_N_Increase_Counter(MatrixIndexes indexes, MatrixValue value, 
			int i, Reporter reporter, CollectMultipleConvertedOutputs collectFinalMultipleOutputs, 
			byte[] resultDimsUnknown, long[] resultsNonZeros, long[] resultsMaxRowDims, 
			long[] resultsMaxColDims) throws IOException
	{
 		collectFinalMultipleOutputs.collectOutput(indexes, value, i, reporter);
		resultsNonZeros[i]+=value.getNonZeros();
		
		if ( resultDimsUnknown[i] == (byte) 1 ) 
		{
			// compute dimensions for the resulting matrix
			
			// find the maximum row index and column index encountered in current output block/cell 
			long maxrow=0, maxcol=0;
		
			try {
				maxrow = value.getMaxRow();
				maxcol = value.getMaxColumn();
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
			
			if ( maxrow > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = maxrow;
				
			if ( maxcol > resultsMaxColDims[i] )
				resultsMaxColDims[i] = maxcol;
		}
		else if(resultDimsUnknown[i] == (byte) 2)
		{
			if ( indexes.getRowIndex() > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = indexes.getRowIndex();
				
			if ( indexes.getColumnIndex() > resultsMaxColDims[i] )
				resultsMaxColDims[i] = indexes.getColumnIndex();
		}
	}

	/**
	 * 
	 * @param mixed_instructions
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	protected void processMixedInstructions(MRInstruction[] mixed_instructions) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( mixed_instructions != null )
			for( MRInstruction ins : mixed_instructions )
				processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
	}
	
	/**
	 * 
	 * @param mixed_instructions
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	protected void processMixedInstructions(Vector<MRInstruction> mixed_instructions) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		if( mixed_instructions != null )
			for( MRInstruction ins : mixed_instructions )
				processOneInstruction(ins, valueClass, cachedValues, tempValue, zeroInput);
	}
	
	/**
	 * 
	 * @param ins
	 * @param valueClass
	 * @param cachedValues
	 * @param tempValue
	 * @param zeroInput
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	protected void processOneInstruction(MRInstruction ins, Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//Timing time = new Timing(true);
		
		if ( ins instanceof AggregateBinaryInstruction ) {
			byte input = ((AggregateBinaryInstruction)ins).input1;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else if(ins instanceof ZeroOutInstruction || ins instanceof AggregateUnaryInstruction 
				|| ins instanceof RangeBasedReIndexInstruction || ins instanceof CumsumSplitInstruction)
		{
			byte input=((UnaryMRInstructionBase) ins).input;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			if( ins instanceof CumsumAggregateInstruction )
				((CumsumAggregateInstruction)ins).setMatrixCharacteristics(dim);
			if( ins instanceof CumsumSplitInstruction )
				((CumsumSplitInstruction)ins).setMatrixCharacteristics(dim);
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else if( ins instanceof ReorgInstruction )
		{
			ReorgInstruction rinst = (ReorgInstruction) ins;
			byte input = rinst.input;
			MatrixCharacteristics dim = dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			rinst.setInputMatrixCharacteristics(dim);
			rinst.setOutputEmptyBlocks(!(this instanceof MMCJMRMapper)); //MMCJMRMapper does not output empty blocks, no need to generate
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else if( ins instanceof MatrixReshapeMRInstruction )
		{
			MatrixReshapeMRInstruction mrins = (MatrixReshapeMRInstruction) ins;
			byte input = mrins.input;
			byte output = mrins.output; 
			MatrixCharacteristics dimIn=dimensions.get(input);
			MatrixCharacteristics dimOut=dimensions.get(output);
			if(dimIn==null || dimOut==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			mrins.setMatrixCharacteristics(dimIn, dimOut);
			mrins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dimIn.numRowsPerBlock, dimIn.numColumnsPerBlock);
		}
		else if(ins instanceof AppendMInstruction)
		{
			byte input=((AppendMInstruction) ins).input1;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else if(ins instanceof BinaryMInstruction || ins instanceof RemoveEmptyMRInstruction )
		{
			byte input=((BinaryMRInstructionBase) ins).input1;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else if(ins instanceof AppendGInstruction)
		{
			AppendGInstruction arinst = ((AppendGInstruction) ins);
			byte input = arinst.input1;
			MatrixCharacteristics dimIn=dimensions.get(input);
			if( dimIn==null )
				throw new DMLRuntimeException("Dimensions for instruction "+arinst+"  is unset!!!");
			arinst.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dimIn.numRowsPerBlock, dimIn.numColumnsPerBlock);
		}
		else if(ins instanceof UnaryMRInstructionBase)
		{
			UnaryMRInstructionBase rinst = (UnaryMRInstructionBase) ins;
			MatrixCharacteristics dimIn=dimensions.get(rinst.input);
			if( dimIn==null )
				throw new DMLRuntimeException("Dimensions for instruction "+rinst+"  is unset!!!");
			rinst.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dimIn.numRowsPerBlock, dimIn.numColumnsPerBlock);
		}
		else if(ins instanceof BinaryMRInstructionBase)
		{
			BinaryMRInstructionBase rinst = (BinaryMRInstructionBase) ins;
			MatrixCharacteristics dimIn=dimensions.get(rinst.input1);
			if( dimIn!=null ) //not set for all
				rinst.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dimIn.numRowsPerBlock, dimIn.numColumnsPerBlock);
			else
				ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, -1, -1);
		}
		else
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, -1, -1);
	
		//System.out.println(ins.getMRInstructionType()+" in "+time.stop());
	}
	
	/**
	 * Reset in-memory state from distributed cache (required only for
	 * local job runner) 
	 */
	public static void resetDistCache()
	{
		for(DistributedCacheInput dcInput : dcValues.values() ) 
			dcInput.reset();
		dcValues.clear();
	}

	
}
