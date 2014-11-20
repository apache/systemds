/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.DataGenMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.TaggedMatrixValue;


public abstract class MapperBase extends MRBaseForCommonInstructions
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(MapperBase.class);
	
	//the indexes that this particular input matrix file represents
	protected Vector<Byte> representativeMatrixes=null;
	
	//the dimension for all the representative matrices 
	//(they are all the same, since coming from the same files)
	protected long[] rlens=null;
	protected long[] clens=null;
	
	//the block sizes for the representative matrices
	protected int[] brlens=null;
	protected int[] bclens=null;
	
	//upper boundaries to check
	protected long[] rbounds=null;
	protected long[] cbounds=null;
	
	//boundary block sizes
	protected int[] lastblockrlens=null;
	protected int[] lastblockclens=null;
	
	//rand instructions that need to be performed in mapper
	protected Vector<DataGenMRInstruction> dataGen_instructions=new Vector<DataGenMRInstruction>();
	
	//instructions that need to be performed in mapper
	protected Vector<Vector<MRInstruction>> mapper_instructions=new Vector<Vector<MRInstruction>>();
	
	//block instructions that need to be performed in part by mapper
	protected Vector<Vector<ReblockInstruction>> reblock_instructions=new Vector<Vector<ReblockInstruction>>();
	
	//csv block instructions that need to be performed in part by mapper
	protected Vector<Vector<CSVReblockInstruction>> csv_reblock_instructions=new Vector<Vector<CSVReblockInstruction>>();
	
	//the indexes of the matrices that needed to be outputted
	protected Vector<Vector<Byte>> outputIndexes=new Vector<Vector<Byte>>();
	
	//converter to convert the input record into indexes and matrix value (can be a cell or a block)
	protected Converter inputConverter=null;
	
	//a counter to measure the time spent in a mapper
	protected static enum Counters {
		MAP_TIME 
	};
	
	
	protected void commonMap(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, Reporter reporter) 
		throws IOException 
	{
		long start=System.currentTimeMillis();
		
		//System.out.println("read in Mapper: "+rawKey+": "+rawValue);
		
		//for each representative matrix, read the record and apply instructions
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			byte thisMatrix=representativeMatrixes.get(i);
			
			//convert the record into the right format for the representative matrix
			inputConverter.setBlockSize(brlens[i], bclens[i]);
			inputConverter.convert(rawKey, rawValue);
			
			//apply unary instructions on the converted indexes and values
			while(inputConverter.hasNext())
			{
				Pair<MatrixIndexes, MatrixValue> pair = inputConverter.next();
				
				MatrixIndexes indexes=pair.getKey();
				MatrixValue value=pair.getValue();
				
				checkValidity(indexes, value, i);
				
				//put the input in the cache
				cachedValues.reset();
				cachedValues.set(thisMatrix, indexes, value);
				
				//special operations for individual mapp type
				specialOperationsForActualMap(i, out, reporter);
			}
		}
		
		reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
	}
	
	protected abstract void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)throws IOException;

	protected void checkValidity(MatrixIndexes indexes, MatrixValue value, int rep) throws IOException
	{
		if(indexes.getRowIndex()<=0 || indexes.getColumnIndex()<=0 
		|| indexes.getRowIndex()>rbounds[rep] || indexes.getColumnIndex()>cbounds[rep]){
			
			throw new IOException("key: "+indexes+" is out of range: [1, "+rbounds[rep]+"] and [1, "+cbounds[rep]+"]!");
		}
		
		if(indexes.getRowIndex()==rbounds[rep] && value.getNumRows()>lastblockrlens[rep])
		{
			throw new IOException("boundary block with "+value.getNumRows()+" rows exceeds the size "+lastblockrlens[rep]);
		}
		
		if(indexes.getColumnIndex()==cbounds[rep] && value.getNumColumns()>lastblockclens[rep])
		{
			throw new IOException("boundary block with "+value.getNumColumns()+" columns exceeds the size "+lastblockclens[rep]);
		}
	}

	private void setupDistCacheFiles(JobConf job, long[] rlens, long[] clens, int[] brlens, int[] bclens) throws IOException {
		
		if ( MRJobConfiguration.getDistCacheInputIndices(job) == null )
			return;
		
		//boolean isJobLocal = false;
		isJobLocal = InfrastructureAnalyzer.isLocalMode(job);
		
		String[] inputIndices = MRJobConfiguration.getInputPaths(job);
		String[] dcIndices = MRJobConfiguration.getDistCacheInputIndices(job).split(Instruction.INSTRUCTION_DELIM);
		Path[] dcFiles = DistributedCache.getLocalCacheFiles(job);
		PDataPartitionFormat[] inputPartitionFormats = MRJobConfiguration.getInputPartitionFormats(job);
		
		DistributedCacheInput[] dcInputs = new DistributedCacheInput[dcIndices.length];
		for(int i=0; i < dcIndices.length; i++) {
        	byte inputIndex = Byte.parseByte(dcIndices[i]);
        	
        	//load if not already present (jvm reuse)
        	if( !dcValues.containsKey(inputIndex) )
        	{
				// When the job is in local mode, files can be read from HDFS directly -- use 
				// input paths as opposed to "local" paths prepared by DistributedCache. 
	        	Path p = null;
				if(isJobLocal)
					p = new Path(inputIndices[ Byte.parseByte(dcIndices[i]) ]);
				else
					p = dcFiles[i];
				
				dcInputs[i] = new DistributedCacheInput(
									p, 
									MRJobConfiguration.getNumRows(job, inputIndex), //rlens[inputIndex],
									MRJobConfiguration.getNumColumns(job, inputIndex), //clens[inputIndex],
									MRJobConfiguration.getNumRowsPerBlock(job, inputIndex), //brlens[inputIndex],
									MRJobConfiguration.getNumColumnsPerBlock(job, inputIndex), //bclens[inputIndex],
									inputPartitionFormats[inputIndex]
								);
	        	dcValues.put(inputIndex, dcInputs[i]);
        	}
		}
		
	}

	/**
	 * Determines if empty blocks can be discarded on map input. Conceptually, this is true
	 * if the individual instruction don't need to output empty blocks and if they are sparsesafe.
	 * 
	 * @return
	 */
	public boolean allowsFilterEmptyInputBlocks()
	{
		boolean ret = true;
		int count = 0;
		
		if( ret && mapper_instructions!=null )
			for( Vector<MRInstruction> vinst : mapper_instructions )
				for( MRInstruction inst : vinst ){
					ret &= (inst instanceof AggregateBinaryInstruction 
						   && !((AggregateBinaryInstruction)inst).getOutputEmptyBlocks() );
					count++; //ensure that mapper instructions exists
				}
		
		return ret && count>0;
	}
	
	public void configure(JobConf job)
	{
		super.configure(job);
		
		//get the indexes that this matrix file represents, 
		//since one matrix file can occur multiple times in a statement
		try {
			representativeMatrixes=MRJobConfiguration.getInputMatrixIndexesInMapper(job);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, representativeMatrixes.get(0));
		
		DataGenMRInstruction[] allDataGenIns;
		MRInstruction[] allMapperIns;
		ReblockInstruction[] allReblockIns;
		CSVReblockInstruction[] allCSVReblockIns;
		
		try {
			allDataGenIns = MRJobConfiguration.getDataGenInstructions(job);
			
			//parse the instructions on the matrices that this file represent
			allMapperIns=MRJobConfiguration.getInstructionsInMapper(job);
			
			//parse the reblock instructions on the matrices that this file represent
			allReblockIns=MRJobConfiguration.getReblockInstructions(job);
			
			allCSVReblockIns=MRJobConfiguration.getCSVReblockInstructions(job);
			
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		//get all the output indexes
		byte[] outputs=MRJobConfiguration.getOutputIndexesInMapper(job);
		
		//get the dimension of all the representative matrices
		rlens=new long[representativeMatrixes.size()];
		clens=new long[representativeMatrixes.size()];
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			rlens[i]=MRJobConfiguration.getNumRows(job, representativeMatrixes.get(i));
			clens[i]=MRJobConfiguration.getNumColumns(job, representativeMatrixes.get(i));
		//	System.out.println("get dimension for "+representativeMatrixes.get(i)+": "+rlens[i]+", "+clens[i]);
		}
		
		//get the block sizes of the representative matrices
		brlens=new int[representativeMatrixes.size()];
		bclens=new int[representativeMatrixes.size()];
		
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			brlens[i]=MRJobConfiguration.getNumRowsPerBlock(job, representativeMatrixes.get(i));
			bclens[i]=MRJobConfiguration.getNumColumnsPerBlock(job, representativeMatrixes.get(i));
		//	System.out.println("get blocksize for "+representativeMatrixes.get(i)+": "+brlens[i]+", "+bclens[i]);
		}
		
		rbounds=new long[representativeMatrixes.size()];
		cbounds=new long[representativeMatrixes.size()];
		
		lastblockrlens=new int[representativeMatrixes.size()];
		lastblockclens=new int[representativeMatrixes.size()];
		//calculate upper boundaries for key value pairs
		if(valueClass.equals(MatrixBlock.class))
		{
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				rbounds[i]=(long)Math.ceil((double)rlens[i]/(double)brlens[i]);
				cbounds[i]=(long)Math.ceil((double)clens[i]/(double)bclens[i]);
				
				lastblockrlens[i]=(int) (rlens[i]%brlens[i]);
				lastblockclens[i]=(int) (clens[i]%bclens[i]);
				if(lastblockrlens[i]==0)
					lastblockrlens[i]=brlens[i];
				if(lastblockclens[i]==0)
					lastblockclens[i]=bclens[i];
				
				/*
				 * what is this for????
				// DRB: the row indexes need to be fixed 
				rbounds[i] = rlens[i];*/
			}
		}else
		{
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				rbounds[i]=rlens[i];
				cbounds[i]=clens[i];
				lastblockrlens[i]=1;
				lastblockclens[i]=1;
			//	System.out.println("get bound for "+representativeMatrixes.get(i)+": "+rbounds[i]+", "+cbounds[i]);
			}
		}
				
		//load data from distributed cache (if required, reuse if jvm_reuse)
		try
		{
			setupDistCacheFiles(job, rlens, clens, brlens, bclens);
		}
		catch(IOException ex)
		{
			throw new RuntimeException(ex);
		}

		//collect unary instructions for each representative matrix
		HashSet<Byte> set=new HashSet<Byte>();
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			set.clear();
			set.add(representativeMatrixes.get(i));
			
			//collect the relavent datagen instructions for this representative matrix
			Vector<DataGenMRInstruction> dataGensForThisMatrix=new Vector<DataGenMRInstruction>();
			if(allDataGenIns!=null)
			{
				for(DataGenMRInstruction ins:allDataGenIns)
				{
					if(set.contains(ins.input))
					{
						dataGensForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
			if(dataGensForThisMatrix.size()>1)
				throw new RuntimeException("only expects at most one rand instruction per input");
			if(dataGensForThisMatrix.isEmpty())
				dataGen_instructions.add(null);
			else
				dataGen_instructions.add(dataGensForThisMatrix.firstElement());
						
			//collect the relavent instructions for this representative matrix
			Vector<MRInstruction> opsForThisMatrix=new Vector<MRInstruction>();
			
			if(allMapperIns!=null)
			{
				for(MRInstruction ins: allMapperIns)
				{
					try {
						/*
						boolean toAdd=true;
						for(byte input: ins.getInputIndexes())
							if(!set.contains(input))
							{
								toAdd=false;
								break;
							}
							*/
						boolean toAdd=false;
						for(byte input : ins.getInputIndexes())
							if(set.contains(input))
							{
								toAdd=true;
								break;
							}
						
						if(toAdd)
						{
							opsForThisMatrix.add(ins);
							set.add(ins.output);
						}
					} catch (DMLRuntimeException e) {
						throw new RuntimeException(e);
					}	
				}
			}
			
			mapper_instructions.add(opsForThisMatrix);
			
			//collect the relavent reblock instructions for this representative matrix
			Vector<ReblockInstruction> reblocksForThisMatrix=new Vector<ReblockInstruction>();
			if(allReblockIns!=null)
			{
				for(ReblockInstruction ins:allReblockIns)
				{
					if(set.contains(ins.input))
					{
						reblocksForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
		
			reblock_instructions.add(reblocksForThisMatrix);
			
			//collect the relavent reblock instructions for this representative matrix
			Vector<CSVReblockInstruction> csvReblocksForThisMatrix=new Vector<CSVReblockInstruction>();
			if(allCSVReblockIns!=null)
			{
				for(CSVReblockInstruction ins:allCSVReblockIns)
				{
					if(set.contains(ins.input))
					{
						csvReblocksForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
		
			csv_reblock_instructions.add(csvReblocksForThisMatrix);
			
			//collect the output indexes for this representative matrix
			Vector<Byte> outsForThisMatrix=new Vector<Byte>();
			for(byte output: outputs)
			{
				if(set.contains(output))
					outsForThisMatrix.add(output);
			}
			outputIndexes.add(outsForThisMatrix);
		}
	}

	protected void processMapperInstructionsForMatrix(int index) 
		throws IOException
	{
		//apply all mapper instructions
		try {
			processMixedInstructions(mapper_instructions.get(index));
		} catch (Exception e) {
			throw new IOException(e);
		}
	}
	
	protected void processMapOutputToReducer(int index, MatrixIndexes indexBuffer, 
			TaggedMatrixValue taggedValueBuffer, OutputCollector<Writable, Writable> out) throws IOException
	{
			
		for(byte output: outputIndexes.get(index))
		{
			ArrayList<IndexedMatrixValue> results= cachedValues.get(output);
			if(results==null)
				continue;
			for(IndexedMatrixValue result: results)
			{
				if(result==null)
					continue;
				indexBuffer.setIndexes(result.getIndexes());
				////////////////////////////////////////
			//	taggedValueBuffer.getBaseObject().copy(result.getValue());
				taggedValueBuffer.setBaseObject(result.getValue());
				////////////////////////////////////////
				taggedValueBuffer.setTag(output);
				out.collect(indexBuffer, taggedValueBuffer);
			//	System.out.println("map output: "+indexBuffer+"\n"+taggedValueBuffer);
			}
			
		}	
	}
}
