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
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.instructions.mr.PMMJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.data.TaggedMatrixValue;

@SuppressWarnings("rawtypes")
public abstract class MapperBase extends MRBaseForCommonInstructions
{
	
	protected static final Log LOG = LogFactory.getLog(MapperBase.class);
	
	//the indexes that this particular input matrix file represents
	protected ArrayList<Byte> representativeMatrixes=null;
	
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
	protected ArrayList<DataGenMRInstruction> dataGen_instructions=new ArrayList<DataGenMRInstruction>();
	
	//instructions that need to be performed in mapper
	protected ArrayList<ArrayList<MRInstruction>> mapper_instructions=new ArrayList<ArrayList<MRInstruction>>();
	
	//block instructions that need to be performed in part by mapper
	protected ArrayList<ArrayList<ReblockInstruction>> reblock_instructions=new ArrayList<ArrayList<ReblockInstruction>>();
	
	//csv block instructions that need to be performed in part by mapper
	protected ArrayList<ArrayList<CSVReblockInstruction>> csv_reblock_instructions=new ArrayList<ArrayList<CSVReblockInstruction>>();
	
	//the indexes of the matrices that needed to be outputted
	protected ArrayList<ArrayList<Byte>> outputIndexes=new ArrayList<ArrayList<Byte>>();
	
	//converter to convert the input record into indexes and matrix value (can be a cell or a block)
	protected Converter inputConverter=null;
	
	//a counter to measure the time spent in a mapper
	protected static enum Counters {
		MAP_TIME 
	};
	
	
	@SuppressWarnings("unchecked")
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
			
			throw new IOException("key: "+indexes+" is out of range: [1, "+rbounds[rep]+"] and [1, "+cbounds[rep]+"] (tag="+rep+")!");
		}
		
		if(indexes.getRowIndex()==rbounds[rep] && value.getNumRows()>lastblockrlens[rep])
		{
			throw new IOException("boundary block with "+value.getNumRows()+" rows exceeds the size "+lastblockrlens[rep]+" "
					+ "(tag="+rep+", ix="+indexes+", "+value.getNumRows()+"x"+value.getNumColumns()+")");
		}
		
		if(indexes.getColumnIndex()==cbounds[rep] && value.getNumColumns()>lastblockclens[rep])
		{
			throw new IOException("boundary block with "+value.getNumColumns()+" columns exceeds the size "+lastblockclens[rep]+" "
					+ "(tag="+rep+", ix="+indexes+", "+value.getNumRows()+"x"+value.getNumColumns()+")");
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
			for( ArrayList<MRInstruction> vinst : mapper_instructions )
				for( MRInstruction inst : vinst ){
					ret &= (inst instanceof AggregateBinaryInstruction && !((AggregateBinaryInstruction)inst).getOutputEmptyBlocks() )
						  ||(inst instanceof PMMJMRInstruction && !((PMMJMRInstruction)inst).getOutputEmptyBlocks() );
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
		try {
			setupDistCacheFiles(job);
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
			ArrayList<DataGenMRInstruction> dataGensForThisMatrix=new ArrayList<DataGenMRInstruction>();
			if(allDataGenIns!=null)
			{
				for(DataGenMRInstruction ins:allDataGenIns)
				{
					if(set.contains(ins.getInput()))
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
				dataGen_instructions.add(dataGensForThisMatrix.get(0));
						
			//collect the relavent instructions for this representative matrix
			ArrayList<MRInstruction> opsForThisMatrix=new ArrayList<MRInstruction>();
			
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
			ArrayList<ReblockInstruction> reblocksForThisMatrix=new ArrayList<ReblockInstruction>();
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
			ArrayList<CSVReblockInstruction> csvReblocksForThisMatrix=new ArrayList<CSVReblockInstruction>();
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
			ArrayList<Byte> outsForThisMatrix=new ArrayList<Byte>();
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
