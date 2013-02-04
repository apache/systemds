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

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.TaggedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.TaggedPartialBlock;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public abstract class MapperBase extends MRBaseForCommonInstructions{
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
	protected Vector<RandInstruction> rand_instructions=new Vector<RandInstruction>();
	
	//instructions that need to be performed in mapper
	protected Vector<Vector<MRInstruction>> mapper_instructions=new Vector<Vector<MRInstruction>>();
	
	//block instructions that need to be performed in part by mapper
	protected Vector<Vector<ReblockInstruction>> reblock_instructions=new Vector<Vector<ReblockInstruction>>();
	
	//the indexes of the matrices that needed to be outputted
	protected Vector<Vector<Byte>> outputIndexes=new Vector<Vector<Byte>>();
	
	//converter to convert the input record into indexes and matrix value (can be a cell or a block)
	protected Converter inputConverter=null;
	
	//a counter to measure the time spent in a mapper
	protected static enum Counters {MAP_TIME };
	
	protected void commonMap(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, 
	Reporter reporter) throws IOException 
	{
		long start=System.currentTimeMillis();
		
		//System.out.println("read in Mapper: "+rawKey+": "+rawValue);
		
		//for each represenattive matrix, read the record and apply instructions
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			//convert the record into the right format for the representatice matrix
			inputConverter.setBlockSize(brlens[i], bclens[i]);
			inputConverter.convert(rawKey, rawValue);
			
			byte thisMatrix=representativeMatrixes.get(i);
			
			//apply unary instructions on the converted indexes and values
			while(inputConverter.hasNext())
			{
				Pair<MatrixIndexes, MatrixValue> pair=inputConverter.next();
			//	System.out.println("convert to: "+pair);
				MatrixIndexes indexes=pair.getKey();
				
				MatrixValue value=pair.getValue();
				
			//	System.out.println("after converter: "+indexes+" -- "+value);
				
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

	private void loadDistCacheFiles(JobConf job, long[] rlens, long[] clens) throws IOException {
		
		if ( MRJobConfiguration.getDistCacheInputIndices(job) == null )
			return;
		
		//boolean isJobLocal = false;
		if(job.get("mapred.job.tracker").equalsIgnoreCase("local")) {
			isJobLocal = true;
		}
		else {
			isJobLocal = false;
		}

		String[] indices = MRJobConfiguration.getDistCacheInputIndices(job).split(Instruction.INSTRUCTION_DELIM);
		distCacheFiles = DistributedCache.getLocalCacheFiles(job);
		inputPartitionFlags   = MRJobConfiguration.getInputPartitionFlags(job);
		inputPartitionFormats = MRJobConfiguration.getInputPartitionFormats(job);
		inputPartitionSizes   = MRJobConfiguration.getInputPartitionSizes(job);
		
		if ( isJobLocal ) {
			// When the job is in local mode, files can be read from HDFS directly -- use 
			// input paths as opposed to "local" paths prepared by DistributedCache. 
			String[] inputs = MRJobConfiguration.getInputPaths(job);
			for(int i=0; i < indices.length; i++) {
				distCacheFiles[i] = new Path(inputs[ Byte.parseByte(indices[i]) ]);
			}
		}
		
		if ( distCacheFiles.length != indices.length ) {
			throw new IOException("Unexpected error in loadDistCacheFiles(). #Cachefiles (" + distCacheFiles.length + ") != #indices (" + indices.length + ")");
		}
		
		distCacheIndices = new byte[distCacheFiles.length];
		distCacheNumRows = new long[distCacheFiles.length];
		distCacheNumColumns = new long[distCacheFiles.length];
		
		// Load data from distributed cache only if the data is NOT partitioned!!
		if (null != distCacheFiles && distCacheFiles.length > 0 ) {
			for(int i=0; i < distCacheFiles.length; i++) {
	        	byte index = Byte.parseByte(indices[i]);
	        	distCacheIndices[i] = index;
		        distCacheNumRows[i] = MRJobConfiguration.getNumRows(job, index);
		        distCacheNumColumns[i] = MRJobConfiguration.getNumColumns(job, index);
		        
	        	/*if ( inputPartitionFlags[index] == false ) {
		        	Path cachePath = distCacheFiles[i];
		        	
		        	LOG.trace("Reading cached file " + cachePath.getName() + ", " + cachePath.toString() + " from " + (isJobLocal ? "HDFS" : "LOCAL-FS"));
		        	//System.out.println("Reading cached file " + cachePath.getName() + ", " + cachePath.toString() + " from " + (isJobLocal ? "HDFS" : "LOCAL-FS"));
		        	long st = System.currentTimeMillis();
		        	MatrixBlock data = DataConverter.readMatrixFromHDFS(
		        			cachePath.toString(), InputInfo.BinaryBlockInputInfo, 
		        			MRJobConfiguration.getNumRows(job, index), // use rlens 
		        			MRJobConfiguration.getNumColumns(job, index), 
		        			MRJobConfiguration.getNumRowsPerBlock(job, index), 
		        			MRJobConfiguration.getNumColumnsPerBlock(job, index), 1.0, !isJobLocal);
		        	LOG.trace("reading from dist cache complete.." + data.getNumRows() + ", "+ data.getNumColumns() + ": " + (System.currentTimeMillis()-st) + " msec");
		        	//System.out.println("reading from dist cache complete.." + data.getNumRows() + ", "+ data.getNumColumns() + ": " + (System.currentTimeMillis()-st) + " msec");
		        	distCacheValues.put(index, data);
		        }
		        else {
		        	LOG.trace("Postponed reading of cached file " + distCacheFiles[i] + " from " + (isJobLocal ? "HDFS" : "LOCAL-FS") + ", as it is partitioned (format=" + inputPartitionFormats[index] + ")");		        
		        }*/
			}
		}
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
		
		RandInstruction[] allRandIns;
		MRInstruction[] allMapperIns;
		ReblockInstruction[] allReblockIns;
		try {
			allRandIns = MRJobConfiguration.getRandInstructions(job);
			
			//parse the instructions on the matrices that this file represent
			allMapperIns=MRJobConfiguration.getInstructionsInMapper(job);
			
			//parse the reblock instructions on the matrices that this file represent
			allReblockIns=MRJobConfiguration.getReblockInstructions(job);
			
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
				
		try {
			loadDistCacheFiles(job, rlens, clens);
		} catch (IOException e) {
			e.printStackTrace();
		}

		//collect unary instructions for each representative matrix
		HashSet<Byte> set=new HashSet<Byte>();
		for(int i=0; i<representativeMatrixes.size(); i++)
		{
			set.clear();
			set.add(representativeMatrixes.get(i));
			
			//collect the relavent rand instructions for this representative matrix
			Vector<RandInstruction> randsForThisMatrix=new Vector<RandInstruction>();
			if(allRandIns!=null)
			{
				for(RandInstruction ins:allRandIns)
				{
					if(set.contains(ins.input))
					{
						randsForThisMatrix.add(ins);
						set.add(ins.output);
					}
				}
			}
			if(randsForThisMatrix.size()>1)
				throw new RuntimeException("only expects at most one rand instruction per input");
			if(randsForThisMatrix.isEmpty())
				rand_instructions.add(null);
			else
				rand_instructions.add(randsForThisMatrix.firstElement());
						
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
	
	protected void processReblockInMapperAndOutput(int index, MatrixIndexes indexBuffer,
			TaggedPartialBlock partialBuffer, OutputCollector<Writable, Writable> out) 
	throws IOException
	{
		//apply reblock instructions
		for(ReblockInstruction ins: reblock_instructions.get(index))
		{
			IndexedMatrixValue inValue=cachedValues.getFirst(ins.input);
			if(inValue==null)
				continue;
			long bi=UtilFunctions.blockIndexCalculation(inValue.getIndexes().getRowIndex(),ins.brlen);
			long bj=UtilFunctions.blockIndexCalculation(inValue.getIndexes().getColumnIndex(),ins.bclen);
			indexBuffer.setIndexes(bi, bj);
			
			int ci=UtilFunctions.cellInBlockCalculation(inValue.getIndexes().getRowIndex(), ins.brlen);
			int cj=UtilFunctions.cellInBlockCalculation(inValue.getIndexes().getColumnIndex(),ins.bclen);
			partialBuffer.getBaseObject().set(ci, cj, ((MatrixCell)inValue.getValue()).getValue());
			partialBuffer.setTag(ins.output);
			out.collect(indexBuffer, partialBuffer);
		//	System.out.println("in Mapper, "+inValue+" --> "+indexBuffer+": "+partialBuffer);
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
