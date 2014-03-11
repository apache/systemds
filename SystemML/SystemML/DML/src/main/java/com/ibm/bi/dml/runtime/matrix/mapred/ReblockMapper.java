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
import java.util.Map.Entry;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.AdaptivePartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedAdaptivePartialBlock;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

/**
 * 
 * 
 */
public class ReblockMapper extends MapperBase 
	implements Mapper<Writable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//state of reblock mapper
	private boolean outputDummyRecords = false;
	private OutputCollector<Writable, Writable> cachedCollector = null;
	private HashMap<Byte, MatrixCharacteristics> dimensionsOut = new HashMap<Byte, MatrixCharacteristics>();
	private HashMap<Byte, MatrixCharacteristics> dimensionsIn = new HashMap<Byte, MatrixCharacteristics>();
	
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
		outputDummyRecords = MapReduceTool.getUniqueKeyPerTask(job, true).equals("0");
		
		try 
		{
			ReblockInstruction[] reblockInstructions = MRJobConfiguration.getReblockInstructions(job);
		
			//get dimension information
			for(ReblockInstruction ins: reblockInstructions)
			{
				dimensionsIn.put(ins.input, MRJobConfiguration.getMatrixCharacteristicsForInput(job, ins.input));
				dimensionsOut.put(ins.output, MRJobConfiguration.getMatrixCharactristicsForReblock(job, ins.output));
			}
		
			//compute reblock buffer size (according to relevant rblk inst of this task only)
			buffersize = ReblockBuffer.DEFAULT_BUFFER_SIZE/reblock_instructions.size();
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
		
		//handle empty block output (on first task)
		if( !outputDummyRecords || cachedCollector==null )
			return;
		
		MatrixIndexes tmpIx = new MatrixIndexes();
		TaggedAdaptivePartialBlock tmpVal = new TaggedAdaptivePartialBlock();
		AdaptivePartialBlock apb = new AdaptivePartialBlock(new PartialBlock(-1,-1,0));
		tmpVal.setBaseObject(apb);
		for(Entry<Byte, MatrixCharacteristics> e: dimensionsOut.entrySet())
		{
			tmpVal.setTag(e.getKey());
			MatrixCharacteristics mc = e.getValue();
			long rlen = mc.numRows;
			long clen = mc.numColumns;
			int brlen = mc.numRowsPerBlock;
			int bclen = mc.numColumnsPerBlock;
			long nnz = mc.nonZero;
			
			
			//output empty blocks on demand (not required if nnz ensures dense matrix)
			if( nnz < (rlen*clen-Math.min(brlen, rlen)*Math.min(bclen, clen)+1) )
				for(long i=0, r=1; i<rlen; i+=brlen, r++)
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
						rbuff = new ReblockBuffer( buffersize, mc.get_rows(), mc.get_cols(), ins.brlen, ins.bclen );
						buffer.put(ins.output, rbuff);
					}
					
					//append cells and flush buffer if required
					MatrixValue mval = inValue.getValue();
					if( mval instanceof MatrixBlock )
					{
						MatrixIndexes inIx = inValue.getIndexes();
						MatrixCharacteristics mc = dimensionsIn.get(ins.input);
						long row_offset = (inIx.getRowIndex()-1)*mc.get_rows_per_block() + 1;
						long col_offset = (inIx.getColumnIndex()-1)*mc.get_cols_per_block() + 1;
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
