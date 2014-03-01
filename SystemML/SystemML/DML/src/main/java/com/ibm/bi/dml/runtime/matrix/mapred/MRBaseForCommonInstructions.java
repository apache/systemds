/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendMInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MatrixReshapeMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReorgInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.util.DataConverter;


public class MRBaseForCommonInstructions extends MapReduceBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//indicate whether the matrix value in this mapper is a matrix cell or a matrix block
	protected Class<? extends MatrixValue> valueClass;
	
	//a cache to hold the intermediate results
	protected CachedValueMap cachedValues=new CachedValueMap();
	
	//distributed cache data handling
	public static boolean isJobLocal = false; //set from MapperBase
	public static HashMap<Byte, IndexedMatrixValue> distCacheValues = new HashMap<Byte,IndexedMatrixValue>();
	public static HashMap<Byte, IndexedMatrixValue[]> distCacheValues2 = new HashMap<Byte,IndexedMatrixValue[]>();	
	public static byte[] distCacheIndices = null;
	public static Path[] distCacheFiles = null;
	public static long[] distCacheNumRows = null;
	public static long[] distCacheNumColumns = null;	
	public static boolean[] inputPartitionFlags = null;
	public static PDataPartitionFormat[] inputPartitionFormats = null;
	public static int[] inputPartitionSizes = null;
 	
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
		//TODO: remove redundant code
		//System.out.println("output "+i+", "+indexes+"\n"+value);
		//LOG.info("~~ output: "+indexes+"\n"+value);
		if ( resultDimsUnknown[i] == (byte) 1 ) {
			// compute dimensions for the resulting matrix
			
			// find the maximum row index and column index encountered in current output block/cell 
			long maxrow=0, maxcol=0;
		/*	try {
				maxrow = UtilFunctions.cellIndexCalculation( cachedValues.get(resultIndexes[i]).getIndexes().getRowIndex(), 
						cachedValues.get(resultIndexes[i]).getValue().getNumRows(), cachedValues.get(resultIndexes[i]).getValue().getMaxRow() );
				
				maxcol = UtilFunctions.cellIndexCalculation( cachedValues.get(resultIndexes[i]).getIndexes().getColumnIndex(), 
						cachedValues.get(resultIndexes[i]).getValue().getNumColumns(), cachedValues.get(resultIndexes[i]).getValue().getMaxColumn() );
			} catch(DMLRuntimeException e) {
				e.printStackTrace();
			}*/
			try {
				maxrow = value.getMaxRow();
				maxcol = value.getMaxColumn();
				//System.out.println("maxrow = " + maxrow + ", maxcol = " + maxcol + ", val = " + value.getValue((int)indexes.getRowIndex(), (int)indexes.getColumnIndex()));
			} catch (DMLRuntimeException e) {
				throw new IOException(e);
			}
			
			if ( maxrow > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = maxrow;
				
			if ( maxcol > resultsMaxColDims[i] )
				resultsMaxColDims[i] = maxcol;
		}else if(resultDimsUnknown[i] == (byte) 2)
		{
			if ( indexes.getRowIndex() > resultsMaxRowDims[i] )
				resultsMaxRowDims[i] = indexes.getRowIndex();
				
			if ( indexes.getColumnIndex() > resultsMaxColDims[i] )
				resultsMaxColDims[i] = indexes.getColumnIndex();
			//System.out.println("i = " + i + ", maxrow = " + resultsMaxRowDims[i] + ", maxcol = " + resultsMaxColDims[i] + ", val = " + value.getValue((int)indexes.getRowIndex(), (int)indexes.getColumnIndex()));
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
		if(mixed_instructions==null)
			return;
		for(MRInstruction ins: mixed_instructions)
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
		if(mixed_instructions==null || mixed_instructions.isEmpty())
			return;
		for(MRInstruction ins: mixed_instructions)
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
		
		if(ins instanceof ZeroOutInstruction || ins instanceof AggregateUnaryInstruction 
				|| ins instanceof RangeBasedReIndexInstruction)
		{
			byte input=((UnaryMRInstructionBase) ins).input;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
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
		else if(ins instanceof AppendRInstruction)
		{
			AppendRInstruction arinst = ((AppendRInstruction) ins);
			byte input = arinst.input1;
			MatrixCharacteristics dimIn=dimensions.get(input);
			if( dimIn==null )
				throw new DMLRuntimeException("Dimensions for instruction "+arinst+"  is unset!!!");
			arinst.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dimIn.numRowsPerBlock, dimIn.numColumnsPerBlock);
		}
		else if ( ins instanceof AggregateBinaryInstruction ) {
			byte input = ((AggregateBinaryInstruction)ins).input1;
			MatrixCharacteristics dim=dimensions.get(input);
			if(dim==null)
				throw new DMLRuntimeException("dimension for instruction "+ins+"  is unset!!!");
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, dim.numRowsPerBlock, dim.numColumnsPerBlock);
		}
		else
			ins.processInstruction(valueClass, cachedValues, tempValue, zeroInput, -1, -1);
	
		//System.out.println(ins.getMRInstructionType()+" in "+time.stop());
	}
	
	
	/////////////////////////////////
	// Distributed Cache Handling
	/////////////////////////////////
	
	/**
	 * Reset in-memory state from distributed cache (required only for
	 * local job runner) 
	 */
	public static void resetDistCache()
	{
		distCacheValues.clear();
		distCacheValues2.clear();
		distCacheIndices = null;
		distCacheFiles = null;
		distCacheNumRows = null;
		distCacheNumColumns = null;	
		inputPartitionFlags = null;
		inputPartitionFormats = null;
		inputPartitionSizes = null;
	}

	/**
	 * 
	 * @param input
	 * @param rowBlockIndex
	 * @param colBlockIndex
	 * @param rowBlockSize
	 * @param colBlockSize
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static IndexedMatrixValue getDataFromDistributedCache(byte input, long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//reuse in-memory matrix blocks
		if( distCacheValues2.containsKey(input) )
			return distCacheValues2.get(input)[(int)rowBlockIndex-1];
		
		//load data from distributed cache (cache if full vector)
		IndexedMatrixValue ret = readBlockFromDistributedCache(input, rowBlockIndex, 1, rowBlockSize, colBlockSize); 
		if ( ret == null )
			throw new DMLRuntimeException("Unexpected: vector read from distcache is null!");				
		return ret;
	}

	private static IndexedMatrixValue readBlockFromDistributedCache(byte input, long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//find cache index for input
		int distCache_index = -1;
		for( int i=0; i<distCacheIndices.length; i++ )
			if(distCacheIndices[i] == input){
				distCache_index = i;
				break;
			}
		//return null, if no cache index found
		if(distCache_index == -1)
			return null;

		IndexedMatrixValue ret = null;
		
		if ( !inputPartitionFlags[input] ) //entire vector
		{
			try
			{
				ArrayList<IndexedMatrixValue> tmp = DataConverter.readMatrixBlocksFromHDFS(
		    			distCacheFiles[distCache_index].toString(), InputInfo.BinaryBlockInputInfo, 
		    			distCacheNumRows[distCache_index], // use rlens 
		    			distCacheNumColumns[distCache_index], 
		    			rowBlockSize,colBlockSize, !isJobLocal ); 
			
				IndexedMatrixValue[] vect = new IndexedMatrixValue[tmp.size()];
				for( IndexedMatrixValue val : tmp ) //ix sort
					vect[(int)val.getIndexes().getRowIndex()-1]=val;
				distCacheValues2.put(input, vect);					
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
			
			ret = distCacheValues2.get(input)[(int)rowBlockIndex-1]; 
		}
		else //partitioned input
		{
			MatrixValue mv = getPartitionFromDistributedCache(input, distCache_index, rowBlockIndex, colBlockIndex, rowBlockSize, colBlockSize);
			
			int part_rl, st, end;
			part_rl = (int) ((rowBlockIndex-1)*rowBlockSize/inputPartitionSizes[input])*inputPartitionSizes[input];
			st = (int) ((rowBlockIndex-1)*rowBlockSize - part_rl);
			end = (int) Math.min(rowBlockIndex*rowBlockSize, distCacheNumRows[distCache_index])-part_rl-1;

			MatrixBlock mb = new MatrixBlock(
					(int)Math.min(rowBlockSize, (distCacheNumRows[distCache_index]-(rowBlockIndex-1)*rowBlockSize)), 
					(int)Math.min(colBlockSize, (distCacheNumColumns[distCache_index]-(colBlockIndex-1)*colBlockSize)), false);
			mb = (MatrixBlock) ((MatrixBlockDSM)mv).sliceOperations(st+1, end+1, 1, 1, mb);
			ret = new IndexedMatrixValue(new MatrixIndexes(rowBlockIndex,colBlockIndex),mb);
		}

		return ret;
	}
	
	private static MatrixValue getPartitionFromDistributedCache(byte input, int distCache_index, long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize) 
		throws DMLRuntimeException 
	{	
		IndexedMatrixValue imv = distCacheValues.get(input);

		int partID = computePartitionID(rowBlockIndex, colBlockIndex, rowBlockSize, colBlockSize, inputPartitionFormats[input], inputPartitionSizes[input]);
		//int cachedPartID = 
		boolean readNewPartition = true;
		if ( imv != null ) {
			MatrixIndexes partIdx = imv.getIndexes();
			
			// cached partition's range (from distCacheValues)
			//int part_st = (int) (partID-1)*inputPartitionSizes[input];
			//int part_end = part_st + (int) Math.min(partID*inputPartitionSizes[input], distCacheNumRows[distCache_index]-part_st)-1;
			
			// requested range
			//int req_st = (int) ((rowBlockIndex-1)*DMLTranslator.DMLBlockSize);
			//int req_end = (int) Math.min(rowBlockIndex*DMLTranslator.DMLBlockSize, distCacheNumRows[distCache_index])-1;
			//if ( req_st < req_end && req_st >= part_st && req_end <= part_end ) {
			//	// requested range can be served from distCacheValues, and no need to load a new partition
			//	readNewPartition = false; 
			//}
			
			int cachedPartID = (int) partIdx.getRowIndex();
			if(partID == cachedPartID || inputPartitionFlags[input] == false)
				readNewPartition = false;
			//System.out.println("reqIndex ["+rowBlockIndex+","+colBlockIndex+"] reqRange [" + req_st + "," + req_end +"]  partRange [" + part_st + "," + part_end + "] ... cachedPart " + cachedPartID + " reqPartID " + partID + " --> " + (readNewPartition ? "ReadNew" : "UseCached"));
		}
		if(imv == null || readNewPartition) {
			MatrixValue data = null;
			MatrixIndexes idx = null;

			// If the input data is not partitioned, read the entire matrix from HDFS.
			// Otherwise, read the required partition
			if(inputPartitionFlags[input] == false) {
				try {
					data = DataConverter.readMatrixFromHDFS(
			    			distCacheFiles[distCache_index].toString(), InputInfo.BinaryBlockInputInfo, 
			    			distCacheNumRows[distCache_index], // use rlens 
			    			distCacheNumColumns[distCache_index], 
			    			rowBlockSize, 
			    			colBlockSize, 1.0, !isJobLocal );
				} catch (IOException e) {
					throw new DMLRuntimeException(e);
				}
				idx = new MatrixIndexes(1,1);
			}
			else { 
				data = DataConverter.readPartitionFromDistCache(
						distCacheFiles[distCache_index].toString(), 
						true, 
						distCacheNumRows[distCache_index], distCacheNumColumns[distCache_index],
						rowBlockSize, colBlockSize,
						partID, inputPartitionSizes[input]);
				idx = new MatrixIndexes(partID,1);
			}
			//System.out.println(".... READ " + idx.toString());
			imv = new IndexedMatrixValue(idx, data);
			distCacheValues.put(input, imv);
		}
		
		return imv.getValue();
	}
	
	private static int computePartitionID(long rowBlockIndex, long colBlockIndex, int rowBlockSize, int colBlockSize, PDataPartitionFormat pformat, int psize) throws DMLRuntimeException {
		int pfile = -1; // partition file ID
		switch(pformat) {
		case NONE:
			return -1;
		case ROW_BLOCK_WISE_N:
			pfile = (int) (((rowBlockIndex-1)*rowBlockSize)/psize) + 1;
			break;
		
		case COLUMN_BLOCK_WISE_N:
			pfile = (int) (((colBlockIndex-1)*colBlockSize)/psize) + 1;
			break;
		
		default:
			throw new DMLRuntimeException("Unexpected partitioning format (" + pformat + ") in readPartitionFromDistCache");
		}
		
		return pfile;
	}
	
}
