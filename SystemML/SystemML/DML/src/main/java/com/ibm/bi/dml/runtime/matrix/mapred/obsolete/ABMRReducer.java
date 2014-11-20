/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.mapred.obsolete;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.Multiply;
import com.ibm.bi.dml.runtime.functionobjects.Plus;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.data.TaggedTripleIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateBinaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;


public class ABMRReducer extends MapReduceBase 
implements Reducer<TaggedTripleIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String AGGREGATE_BINARY_OPERATION_CONFIG="aggregate.binray.operation";
	private AggregateBinaryOperator bin_agg_op=new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), new AggregateOperator(0, Plus.getPlusFnObject()));
	private MatrixBlock leftblock;
	private MatrixBlock rightblock;
	private MatrixBlock resultblock;
	private MatrixBlock aggblock;
	private MatrixIndexes aggIndexes=new MatrixIndexes();
	private int numRowLeft, numColumnLeft;
	private int numRowRight, numColumnRight;
	private boolean rightAvailable=false;
	private boolean leftAvailable=false;
	private boolean aggAvailable=false;
	private TaggedTripleIndexes prevIndexes=new TaggedTripleIndexes(-1, -1, -1, (byte)0);
	private OutputCollector<MatrixIndexes, MatrixBlock> outCollector;
	private boolean firsttime=true;
	static enum Counters { REDUCE_TIME };
	private Reporter cachedReporter=null;
//	private static final Log LOG = LogFactory.getLog(ABMRReducer.class);
	
	@Override
	public void reduce(TaggedTripleIndexes taggedTriple, Iterator<MatrixBlock> blocks,
			OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter report)
			throws IOException {
		long start=System.currentTimeMillis();
		if(firsttime)
		{
			outCollector=out;
			firsttime=false;
			cachedReporter=report;
		}
		
/*		LOG.info("######### prevIndexes");
		LOG.info(prevIndexes);
		LOG.info("######### currentIndexes");
		LOG.info(taggedTriple);
*/
		//perform pairwise aggregate binary, and added to the aggregates
		if( prevIndexes.getThirdIndex()!=-1 && 
			(  prevIndexes.getThirdIndex()!=taggedTriple.getThirdIndex() 
			|| prevIndexes.getFirstIndex()!=taggedTriple.getFirstIndex()
			|| prevIndexes.getSecondIndex()!=taggedTriple.getSecondIndex() ) )
		{
			if(leftAvailable && rightAvailable)
			{
				try {
/*					LOG.info("######### multiply two blocks");
					LOG.info(leftblock);
					LOG.info(rightblock);
*/					
					OperationsOnMatrixValues.performAggregateBinaryIgnoreIndexes(leftblock, rightblock, resultblock, bin_agg_op);
/*					LOG.info("######### result");
					LOG.info(resultblock);
*/					
					//if(resultblock.getNonZeros()>0)
					//{
						if(!aggAvailable)
						{
							aggblock.reset(resultblock.getNumRows(), resultblock.getNumColumns(), resultblock.isInSparseFormat());
							aggAvailable=true;
						}
						
/*						LOG.info("######### adding to");
						LOG.info(aggblock);
*/						
						aggblock.binaryOperationsInPlace( bin_agg_op.aggOp.increOp, resultblock);
/*						LOG.info("######### aggregate");
						LOG.info(aggblock);
*/
					//}
					

					leftblock.reset();
					rightblock.reset();
					leftAvailable=false;
					rightAvailable=false;
					
				} catch (Exception e) {
					throw new IOException(e);
				}
			}
				
		}
		
		//output previous results if needed
		if(prevIndexes.getFirstIndex()!=taggedTriple.getFirstIndex() 
				|| prevIndexes.getSecondIndex()!=taggedTriple.getSecondIndex())
		{
			if(prevIndexes.getFirstIndex()!=-1 && aggAvailable) //&& aggblock.getNonZeros()>0)
			{	
				aggIndexes.setIndexes(prevIndexes.getFirstIndex(), prevIndexes.getSecondIndex());
				out.collect(aggIndexes, aggblock);
				
		//		System.out.println("######### output in reducer");
		//		System.out.println(aggIndexes);
/*				if(aggIndexes.getRowIndex()>=100 || aggIndexes.getColumnIndex()>=100)
					throw new IOException("indexes are wrong: "+aggIndexes);*/
			//	LOG.info(aggblock);
		
			}
			
			aggAvailable=false;
		}
		
		if(prevIndexes.getFirstIndex()!=taggedTriple.getFirstIndex() 
				|| prevIndexes.getSecondIndex()!=taggedTriple.getSecondIndex()
				|| prevIndexes.getThirdIndex()!=taggedTriple.getThirdIndex())
		{
			leftblock.reset();
			rightblock.reset();
			leftAvailable=false;
			rightAvailable=false;
		}
	
		//record the current value
		MatrixBlock blockbuffer;
		if(taggedTriple.getTag()==0)
		{
			blockbuffer=leftblock;
//			LOG.info("######### set the left matrix");
			leftAvailable=true;
		}
		else
		{
			blockbuffer=rightblock;
//			LOG.info("######### set the right matrix");
			rightAvailable=true;
		}
		
		try {
			boolean firstIteration=true;
			while(blocks.hasNext())
			{
				MatrixBlock block=blocks.next();
				if(firstIteration)
				{
					blockbuffer.reset(block.getNumRows(), block.getNumColumns(), block.isInSparseFormat());
					
					firstIteration=false;
				}
/*				LOG.info("perform aggregate: ");
				LOG.info("old value\n"+blockbuffer);
*/
				blockbuffer.binaryOperationsInPlace( bin_agg_op.aggOp.increOp, block);
//				LOG.info("new value\n"+blockbuffer);
			}
		} catch (Exception e) {
			throw new IOException(e);
		}
	
		prevIndexes.setIndexes(taggedTriple);
//		LOG.info(blockbuffer);
		report.incrCounter(Counters.REDUCE_TIME, System.currentTimeMillis()-start);
	}

	public void close() throws IOException
	{
		long start=System.currentTimeMillis();
		if(prevIndexes.getFirstIndex()!=-1 && leftAvailable && rightAvailable)
		{
			//perform the last pairwise aggregate binary, and added to the aggregates
			try {
				OperationsOnMatrixValues.performAggregateBinaryIgnoreIndexes(leftblock, rightblock, resultblock, bin_agg_op);
				
/*				LOG.info("######### multiply two blocks");
				LOG.info(leftblock);
				LOG.info(rightblock);
				LOG.info("######### result");
				LOG.info(resultblock);
*/				
			//	if(resultblock.getNonZeros()>0)
			//	{
					if(!aggAvailable)
					{
						aggblock.reset(resultblock.getNumRows(), resultblock.getNumColumns(), resultblock.isInSparseFormat());
						aggAvailable=true;
					}
					
/*					LOG.info("######### adding to");
					LOG.info(aggblock);
*/					
					aggblock.binaryOperationsInPlace( bin_agg_op.aggOp.increOp, resultblock);
					
/*					LOG.info("######### aggregate");
					LOG.info(aggblock);
*/
				//}
				
			} catch (Exception e) {
				throw new IOException(e);
			}
			
		//	if(aggblock.getNonZeros()>0)
		//	{
				aggIndexes.setIndexes(prevIndexes.getFirstIndex(), prevIndexes.getSecondIndex());
				outCollector.collect(aggIndexes, aggblock);
		//		System.out.println("######### output in reducer");
		//		System.out.println(aggIndexes);
		/*		if(aggIndexes.getRowIndex()>=100 || aggIndexes.getColumnIndex()>=100)
					throw new IOException("indexes are wrong: "+aggIndexes);*/
		//		LOG.info(aggblock);
	
		//	}
		}
		if(cachedReporter!=null)
			cachedReporter.incrCounter(Counters.REDUCE_TIME, System.currentTimeMillis()-start);
	}
	
	public void configure(JobConf job)
	{
		String str=job.get(AGGREGATE_BINARY_OPERATION_CONFIG);
		//parse aggregate binary operations
		try {
			bin_agg_op=(AggregateBinaryOperator) ((AggregateBinaryInstruction)AggregateBinaryInstruction.parseInstruction(str)).getOperator();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		numRowLeft=job.getInt(ABMRMapper.BLOCK_NUM_ROW_PREFIX_CONFIG+0, -1);
		numColumnLeft=job.getInt(ABMRMapper.BLOCK_NUM_COLUMN_PREFIX_CONFIG+0, -1);
		numRowRight=job.getInt(ABMRMapper.BLOCK_NUM_ROW_PREFIX_CONFIG+1, -1);
		numColumnRight=job.getInt(ABMRMapper.BLOCK_NUM_COLUMN_PREFIX_CONFIG+1, -1);
		leftblock=new MatrixBlock(numRowLeft, numColumnLeft, false);
		rightblock=new MatrixBlock(numRowRight, numColumnRight, false);
		resultblock=new MatrixBlock(numRowLeft, numColumnRight, false);
		aggblock=new MatrixBlock(numRowLeft, numColumnRight, false);
		
		if(numColumnLeft!=numRowRight)
			throw new RuntimeException("the column of matrix1 does not match the row of matrix 2!");
	}
}
