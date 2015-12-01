/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.AdaptivePartialBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.PartialBlock;
import org.apache.sysml.runtime.matrix.data.TaggedAdaptivePartialBlock;


/**
 * 
 * 
 */
public class ReblockReducer extends ReduceBase 
	implements Reducer<MatrixIndexes, TaggedAdaptivePartialBlock, MatrixIndexes, MatrixBlock>
{
	
	private HashMap<Byte, MatrixCharacteristics> dimensions = new HashMap<Byte, MatrixCharacteristics>();
	
	@Override
	public void reduce(MatrixIndexes indexes, Iterator<TaggedAdaptivePartialBlock> values,
			OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter)
			throws IOException 
	{	
		long start=System.currentTimeMillis();
		
		commonSetup(reporter);
		cachedValues.reset();
		
		//process the reducer part of the reblock operation
		processReblockInReducer(indexes, values, dimensions);
		
		//perform mixed operations
		processReducerInstructions();
		
		//output results
		outputResultsFromCachedValues(reporter);
		
		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	@Override
	public void configure(JobConf job) 
	{
		MRJobConfiguration.setMatrixValueClass(job, true);
		super.configure(job);
		
		try 
		{
			//parse the reblock instructions 
			ReblockInstruction[] reblockInstructions = MRJobConfiguration.getReblockInstructions(job);			
			for(ReblockInstruction ins: reblockInstructions)
				dimensions.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
		} 
		catch(Exception e) 
		{
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * 
	 * @param indexes
	 * @param values
	 * @param dimensions
	 */
	protected void processReblockInReducer(MatrixIndexes indexes, Iterator<TaggedAdaptivePartialBlock> values, 
			HashMap<Byte, MatrixCharacteristics> dimensions)
	{
		while(values.hasNext())
		{
			TaggedAdaptivePartialBlock partial = values.next();
			Byte tag = partial.getTag();
			AdaptivePartialBlock srcBlk = partial.getBaseObject();
			
			//get output block (note: iterator may contain blocks of different output variables)
			IndexedMatrixValue block = cachedValues.getFirst(tag);
			if(block==null )
			{
				MatrixCharacteristics mc = dimensions.get(tag);
				int brlen = mc.getRowsPerBlock();
				int bclen = mc.getColsPerBlock();
				int realBrlen=(int)Math.min((long)brlen, mc.getRows()-(indexes.getRowIndex()-1)*brlen);
				int realBclen=(int)Math.min((long)bclen, mc.getCols()-(indexes.getColumnIndex()-1)*bclen);
				block = cachedValues.holdPlace(tag, valueClass); //sparse block
				block.getValue().reset(realBrlen, realBclen);
				block.getIndexes().setIndexes(indexes);
			}
					
			
			//Timing time = new Timing();
			//time.start();
			
			//merge blocks
			if( srcBlk.isBlocked() ) //BINARY BLOCK
			{			
				try 
				{
					MatrixBlock out = (MatrixBlock)block.getValue(); //always block output
					MatrixBlock in = srcBlk.getMatrixBlock();
					
					out.merge(in, false);
					out.examSparsity();  //speedup subsequent usage
				} 
				catch (DMLRuntimeException e) 
				{
					throw new RuntimeException(e);
				}
			}
			else //BINARY CELL
			{
				MatrixBlock out = (MatrixBlock)block.getValue(); //always block output
				PartialBlock pb = srcBlk.getPartialBlock();
				int row = pb.getRowIndex();
				int column = pb.getColumnIndex();
				if(row>=0 && column >=0) //filter empty block marks
					out.quickSetValue(row, column, pb.getValue());
			}
			
			//System.out.println("Merged block (sparse="+(srcBlk.isBlocked()&&srcBlk.isBlocked()&&!srcBlk.getMatrixBlock().isInSparseFormat())+") in "+time.stop());
			
		}
		
		
	}
}
