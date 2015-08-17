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

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.mr.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.BlockRow;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;

public class CSVReblockReducer extends ReduceBase implements Reducer<TaggedFirstSecondIndexes, BlockRow, MatrixIndexes, MatrixBlock>
{
	
	@Override
	public void reduce(TaggedFirstSecondIndexes key, Iterator<BlockRow> values,
			OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter)
			throws IOException 
	{	
		long start=System.currentTimeMillis();
		
		commonSetup(reporter);
		
		cachedValues.reset();
		
		//process the reducer part of the reblock operation
		processCSVReblock(key, values, dimensions);
		
		//perform mixed operations
		processReducerInstructions();
		
		//output results
		outputResultsFromCachedValues(reporter);
		
		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, System.currentTimeMillis()-start);
	}

	/**
	 * 
	 * @param indexes
	 * @param values
	 * @param dimensions
	 * @throws IOException
	 */
	protected void processCSVReblock(TaggedFirstSecondIndexes indexes, Iterator<BlockRow> values, 
			HashMap<Byte, MatrixCharacteristics> dimensions) throws IOException
	{
		try
		{
			Byte tag=indexes.getTag();
			//there only one block in the cache for this output
			IndexedMatrixValue block=cachedValues.getFirst(tag);
			
			while(values.hasNext())
			{
				BlockRow row=values.next();
				if(block==null)
				{
					block=cachedValues.holdPlace(tag, valueClass);
					int brlen=dimensions.get(tag).getRowsPerBlock();
					int bclen=dimensions.get(tag).getColsPerBlock();
					int realBrlen=(int)Math.min((long)brlen, dimensions.get(tag).getRows()-(indexes.getFirstIndex()-1)*brlen);
					int realBclen=(int)Math.min((long)bclen, dimensions.get(tag).getCols()-(indexes.getSecondIndex()-1)*bclen);
					block.getValue().reset(realBrlen, realBclen, false);
					block.getIndexes().setIndexes(indexes.getFirstIndex(), indexes.getSecondIndex());
				}
				
				MatrixBlock mb = (MatrixBlock) block.getValue();
				mb.copy(row.indexInBlock, row.indexInBlock, 0, row.data.getNumColumns()-1, row.data, false);
			}
			
			((MatrixBlock) block.getValue()).recomputeNonZeros();
		}
		catch(DMLRuntimeException ex)
		{
			throw new IOException(ex);
		}			
	}

	@Override
	public void configure(JobConf job) 
	{
		MRJobConfiguration.setMatrixValueClass(job, true);
		super.configure(job);
		//parse the reblock instructions 
		CSVReblockInstruction[] reblockInstructions;
		try {
			reblockInstructions = MRJobConfiguration.getCSVReblockInstructions(job);
		} catch (DMLUnsupportedOperationException e) {
			throw new RuntimeException(e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		for(ReblockInstruction ins: reblockInstructions)
			dimensions.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
	}
}
