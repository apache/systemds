/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
					int brlen=dimensions.get(tag).numRowsPerBlock;
					int bclen=dimensions.get(tag).numColumnsPerBlock;
					int realBrlen=(int)Math.min((long)brlen, dimensions.get(tag).numRows-(indexes.getFirstIndex()-1)*brlen);
					int realBclen=(int)Math.min((long)bclen, dimensions.get(tag).numColumns-(indexes.getSecondIndex()-1)*bclen);
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
