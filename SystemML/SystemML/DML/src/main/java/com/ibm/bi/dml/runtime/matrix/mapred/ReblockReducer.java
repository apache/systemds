/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.AdaptivePartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.SparseRowsIterator;
import com.ibm.bi.dml.runtime.matrix.io.TaggedAdaptivePartialBlock;


/**
 * 
 * 
 */
public class ReblockReducer extends ReduceBase 
	implements Reducer<MatrixIndexes, TaggedAdaptivePartialBlock, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
				int brlen = mc.numRowsPerBlock;
				int bclen = mc.numColumnsPerBlock;
				int realBrlen=(int)Math.min((long)brlen, mc.numRows-(indexes.getRowIndex()-1)*brlen);
				int realBclen=(int)Math.min((long)bclen, mc.numColumns-(indexes.getColumnIndex()-1)*bclen);
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
					boolean appendOnly = out.isInSparseFormat();
					MatrixBlock in = srcBlk.getMatrixBlock();
					
					//merge copy other blocks
					if( in.isInSparseFormat() ) //SPARSE
					{
						SparseRowsIterator iter = in.getSparseRowsIterator();
						while( iter.hasNext() )
						{
							IJV cell = iter.next();
							if( appendOnly )
								out.appendValue(cell.i, cell.j, cell.v);
							else
								out.quickSetValue(cell.i, cell.j, cell.v);
						}
					}
					else //DENSE
					{
						for( int i=0; i<in.getNumRows(); i++ )
							for( int j=0; j<in.getNumColumns(); j++ )
							{
								double val = in.getValueDenseUnsafe(i, j);
								if( val != 0 ) {
									if( appendOnly )
										out.appendValue(i, j, val);
									else
										out.quickSetValue(i, j, val);
								}
							}
					}
					if( appendOnly )
						out.sortSparseRows();
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
