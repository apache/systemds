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
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.AdaptivePartialBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.PartialBlock;
import org.apache.sysml.runtime.matrix.data.TaggedAdaptivePartialBlock;
import org.apache.sysml.runtime.util.MapReduceTool;

/**
 * 
 * 
 */
public class ReblockMapper extends MapperBase 
	implements Mapper<Writable, Writable, Writable, Writable>
{
	
	//state of reblock mapper
	private OutputCollector<Writable, Writable> cachedCollector = null;
	private JobConf cachedJobConf = null;
	private HashMap<Byte, MatrixCharacteristics> dimensionsOut = new HashMap<Byte, MatrixCharacteristics>();
	private HashMap<Byte, MatrixCharacteristics> dimensionsIn = new HashMap<Byte, MatrixCharacteristics>();
	private HashMap<Byte, Boolean> emptyBlocks = new HashMap<Byte, Boolean>();
	
	//reblock buffer
	private HashMap<Byte, ReblockBuffer> buffer = new HashMap<Byte,ReblockBuffer>();
	private int buffersize =-1;
	
	@Override
	public void map(Writable rawKey, Writable rawValue, OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		cachedCollector = out;
		commonMap(rawKey, rawValue, out, reporter);
	}

	@Override
	public void configure(JobConf job) 
	{
		MRJobConfiguration.setMatrixValueClass(job, false); //worst-case
		super.configure(job);
		
		//cache job conf for use in close 
		cachedJobConf = job;
		
		try 
		{
			ReblockInstruction[] reblockInstructions = MRJobConfiguration.getReblockInstructions(job);
		
			//get dimension information
			for(ReblockInstruction ins: reblockInstructions)
			{
				dimensionsIn.put(ins.input, MRJobConfiguration.getMatrixCharacteristicsForInput(job, ins.input));
				dimensionsOut.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
				emptyBlocks.put(ins.output, ins.outputEmptyBlocks);
			}
		
			//compute reblock buffer size (according to relevant rblk inst of this task only)
			//(buffer size divided by max reblocks per input matrix, because those are shared in JVM)
			int maxlen = 1;
			for( ArrayList<ReblockInstruction> rinst : reblock_instructions )
				maxlen = Math.max(maxlen, rinst.size()); //max reblocks per input				
			buffersize = ReblockBuffer.DEFAULT_BUFFER_SIZE/maxlen;
		} 
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}
	
	
	@Override
	public void close() throws IOException
	{
		super.close();
		
		//flush buffered data
		for( Entry<Byte,ReblockBuffer> e : buffer.entrySet() )
		{
			ReblockBuffer rbuff = e.getValue();
			rbuff.flushBuffer(e.getKey(), cachedCollector);
		}
		
		//handle empty block output (responsibility distributed over all map tasks)
		if( cachedJobConf==null || cachedCollector==null )
			return;
		
		long mapID = Long.parseLong(MapReduceTool.getUniqueKeyPerTask(cachedJobConf, true));
		long numMap = cachedJobConf.getNumMapTasks(); 
		
		MatrixIndexes tmpIx = new MatrixIndexes();
		TaggedAdaptivePartialBlock tmpVal = new TaggedAdaptivePartialBlock();
		AdaptivePartialBlock apb = new AdaptivePartialBlock(new PartialBlock(-1,-1,0));
		tmpVal.setBaseObject(apb);
		for(Entry<Byte, MatrixCharacteristics> e: dimensionsOut.entrySet())
		{
			tmpVal.setTag(e.getKey());
			MatrixCharacteristics mc = e.getValue();
			long rlen = mc.getRows();
			long clen = mc.getCols();
			long brlen = mc.getRowsPerBlock();
			long bclen = mc.getColsPerBlock();
			long nnz = mc.getNonZeros();
			
			//output empty blocks on demand (not required if nnz ensures that values exist in each block)
			if( nnz >= (rlen*clen-Math.min(brlen, rlen)*Math.min(bclen, clen)+1) 
				|| !emptyBlocks.get(e.getKey()) )
			{
				continue; //safe to skip empty block output
			}
			
			//output part of empty blocks (all mappers contribute for better load balance),
			//where mapper responsibility is distributed over row blocks 
			long numBlocks = (long)Math.ceil((double)rlen/brlen);
			long len = (long)Math.ceil((double)numBlocks/numMap);
			long start = mapID * len * brlen;
			long end = Math.min((mapID+1) * len * brlen, rlen);
			for(long i=start, r=start/brlen+1; i<end; i+=brlen, r++)
				for(long j=0, c=1; j<clen; j+=bclen, c++)
				{
					tmpIx.setIndexes(r, c);
					cachedCollector.collect(tmpIx, tmpVal);
				}
		}
	}
	
	@Override
	protected void specialOperationsForActualMap(int index,
			OutputCollector<Writable, Writable> out, Reporter reporter)
		throws IOException 
	{
		//note: invoked from MapperBase for each cell 
		
		//apply all instructions
		processMapperInstructionsForMatrix(index);
		
		//apply reblock instructions and output
		processReblockInMapperAndOutput(index, out);
	}
	
	/**
	 * 
	 * @param index
	 * @param indexBuffer
	 * @param partialBuffer
	 * @param out
	 * @throws IOException
	 */
	protected void processReblockInMapperAndOutput(int index, OutputCollector<Writable, Writable> out) 
		throws IOException
	{
		for(ReblockInstruction ins : reblock_instructions.get(index))
		{
			ArrayList<IndexedMatrixValue> ixvList = cachedValues.get(ins.input);
			if( ixvList!=null ) {
				for(IndexedMatrixValue inValue : ixvList )
				{
					if(inValue==null)
						continue;
					
					//get buffer
					ReblockBuffer rbuff = buffer.get(ins.output);
					if( rbuff==null )
					{
						MatrixCharacteristics mc = dimensionsOut.get(ins.output);
						rbuff = new ReblockBuffer( buffersize, mc.getRows(), mc.getCols(), ins.brlen, ins.bclen );
						buffer.put(ins.output, rbuff);
					}
					
					//append cells and flush buffer if required
					MatrixValue mval = inValue.getValue();
					if( mval instanceof MatrixBlock )
					{
						MatrixIndexes inIx = inValue.getIndexes();
						MatrixCharacteristics mc = dimensionsIn.get(ins.input);
						long row_offset = (inIx.getRowIndex()-1)*mc.getRowsPerBlock() + 1;
						long col_offset = (inIx.getColumnIndex()-1)*mc.getColsPerBlock() + 1;
						//append entire block incl. flush on demand
						rbuff.appendBlock(row_offset, col_offset, (MatrixBlock)mval, ins.output, out );
					}
					else //if( mval instanceof MatrixCell )
					{
						rbuff.appendCell( inValue.getIndexes().getRowIndex(), 
								          inValue.getIndexes().getColumnIndex(), 
								          ((MatrixCell)mval).getValue() );
						
						//flush buffer if necessary
						if( rbuff.getSize() >= rbuff.getCapacity() )
							rbuff.flushBuffer( ins.output, out );		
					}
				}
			}
		}
	}
}
