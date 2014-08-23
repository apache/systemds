/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVWriteReducer.RowBlockForTextOutput;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVWriteReducer.RowBlockForTextOutput.Situation;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class CSVWriteReducer extends ReduceBase implements Reducer<TaggedFirstSecondIndexes, MatrixBlock, NullWritable, RowBlockForTextOutput>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	NullWritable nullKey=NullWritable.get();
	RowBlockForTextOutput outValue=new RowBlockForTextOutput();
	RowBlockForTextOutput zeroBlock=new RowBlockForTextOutput();
	
	long[] rowIndexes=null;
	long[] minRowIndexes=null;
	long[] maxRowIndexes=null;
	long[] colIndexes=null;
	long[] numColBlocks=null;
	int[] colsPerBlock=null;
	int[] lastBlockNCols=null;
	String[] delims=null;
	boolean[] sparses=null;
	boolean firsttime=true;
	int[] tagToResultIndex=null;
	//HashMap<Byte, CSVWriteInstruction> csvWriteInstructions=new HashMap<Byte, CSVWriteInstruction>();
	
	private void addEndingMissingValues(byte tag, Reporter reporter) 
	throws IOException
	{
		long col=colIndexes[tag]+1;
		for(;col<numColBlocks[tag]; col++)
		{
			zeroBlock.numCols=colsPerBlock[tag];
			zeroBlock.setSituation(Situation.MIDDLE);
			collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
		}
		//the last block
		if(col<=numColBlocks[tag])
		{
			zeroBlock.numCols=lastBlockNCols[tag];
			zeroBlock.setSituation(Situation.MIDDLE);
			collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
			colIndexes[tag]=0;
		}
	}
	
	private Situation addMissingRows(byte tag, long stoppingRow, Situation sit, Reporter reporter) throws IOException
	{
		for(long row=rowIndexes[tag]+1; row<stoppingRow; row++)
		{
			for(long c=1; c<numColBlocks[tag]; c++)
			{
				zeroBlock.numCols=colsPerBlock[tag];
				zeroBlock.setSituation(sit);
				collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
				sit=Situation.MIDDLE;
			}
			zeroBlock.numCols=lastBlockNCols[tag];
			zeroBlock.setSituation(sit);
			collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
			colIndexes[tag]=0;
			sit=Situation.NEWLINE;
		}
		colIndexes[tag]=0;
		return sit;
	}
	
	private void addNewlineCharacter(byte tag, Reporter reporter) throws IOException 
	{
		zeroBlock.numCols = 0;
		zeroBlock.setSituation(Situation.NEWLINE);
		collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
	}
	
	@Override
	public void reduce(TaggedFirstSecondIndexes inkey, Iterator<MatrixBlock> inValue,
			OutputCollector<NullWritable, RowBlockForTextOutput> out, Reporter reporter)
			throws IOException 
	{
	
		long begin = System.nanoTime();
		//ReduceBase.prep = ReduceBase.encode = ReduceBase.write = 0;
		if(firsttime)
		{
			cachedReporter=reporter;
			firsttime=false;
		}
		
		byte tag = inkey.getTag();
		zeroBlock.setFormatParameters(delims[tag], sparses[tag]);
		outValue.setFormatParameters(delims[tag], sparses[tag]);
		
		Situation sit = Situation.MIDDLE;
		if(rowIndexes[tag]==minRowIndexes[tag])
			sit=Situation.START;
		else if(rowIndexes[tag]!=inkey.getFirstIndex())
			sit=Situation.NEWLINE;
		
		//check whether need to fill in missing values in previous rows
		if(sit==Situation.NEWLINE)
		{
			//if the previous row has not finished
			addEndingMissingValues(tag, reporter);
		}
		
		if(sit==Situation.NEWLINE||sit==Situation.START)
		{	
			//if a row is completely missing
			sit=addMissingRows(tag, inkey.getFirstIndex(), sit, reporter);
		}
		
		//add missing value at the beginning of this row
		for(long col=colIndexes[tag]+1; col<inkey.getSecondIndex(); col++)
		{
			zeroBlock.numCols=colsPerBlock[tag];
			zeroBlock.setSituation(sit);
			collectFinalMultipleOutputs.directOutput(nullKey, zeroBlock, tagToResultIndex[tag], reporter);
			sit=Situation.MIDDLE;
		}
	
		colIndexes[tag]=inkey.getSecondIndex();
		
		while(inValue.hasNext())
		{
			MatrixBlock block = inValue.next();
			outValue.set(block, sit);
			collectFinalMultipleOutputs.directOutput(nullKey, outValue, tagToResultIndex[tag], reporter);
			resultsNonZeros[tagToResultIndex[tag]] += block.getNonZeros();
			sit=Situation.MIDDLE;
		}
		rowIndexes[tag]=inkey.getFirstIndex();

		reporter.incrCounter(Counters.COMBINE_OR_REDUCE_TIME, (System.nanoTime()-begin));
	}
	
	@Override
	public void configure(JobConf job)
	{
		super.configure(job);
		byte maxIndex=0;
		HashMap<Byte, CSVWriteInstruction> out2Ins=new HashMap<Byte, CSVWriteInstruction>();
		try {
			CSVWriteInstruction[] ins = MRJobConfiguration.getCSVWriteInstructions(job);
			for(CSVWriteInstruction in: ins)
			{
				out2Ins.put(in.output, in);
				if(in.output>maxIndex)
					maxIndex=in.output;
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		int numParitions=job.getNumReduceTasks();
		int taskID=MapReduceTool.getUniqueTaskId(job);
		//LOG.info("## taks id: "+taskID);
		//for efficiency only, the arrays may have missing values
		rowIndexes=new long[maxIndex+1];
		colIndexes=new long[maxIndex+1];
		maxRowIndexes=new long[maxIndex+1];
		minRowIndexes=new long[maxIndex+1];
		int maxCol=0;
		numColBlocks=new long[maxIndex+1];
		lastBlockNCols=new int[maxIndex+1];
		colsPerBlock=new int[maxIndex+1];
		delims=new String[maxIndex+1];
		sparses=new boolean[maxIndex+1];
		tagToResultIndex=new int[maxIndex+1];
		
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte ri=resultIndexes[i];
			tagToResultIndex[ri]=i;
			CSVWriteInstruction in=out2Ins.get(ri);
			MatrixCharacteristics dim=MRJobConfiguration.getMatrixCharacteristicsForInput(job, in.input);
			delims[ri]=in.delim;
			sparses[ri]=in.sparse;
			if(dim.get_cols_per_block()>maxCol)
				maxCol=dim.get_cols_per_block();
			numColBlocks[ri]=(long)Math.ceil((double)dim.get_cols()/(double) dim.get_cols_per_block());
			lastBlockNCols[ri]=(int) (dim.get_cols()%dim.get_cols_per_block());
			colsPerBlock[ri]=dim.get_cols_per_block();
			long rstep=(long)Math.ceil((double)dim.get_rows()/(double)numParitions);
			minRowIndexes[ri]=rowIndexes[ri]=rstep*taskID;
			maxRowIndexes[ri]=Math.min(rstep*(taskID+1), dim.numRows);
			colIndexes[ri]=0;
		}
		
		zeroBlock.container=new double[maxCol];
	}
	
	@Override
	public void close() throws IOException
	{
		for( byte tag : resultIndexes )
		{
			//if the previous row has not finished
			addEndingMissingValues(tag, cachedReporter);
			
			//if a row is completely missing
			addMissingRows(tag, maxRowIndexes[tag]+1, Situation.NEWLINE, cachedReporter); 
			
			// add a newline character at the end of file
			addNewlineCharacter(tag, cachedReporter);
		}
		
		super.close();
	}

	

	public static class RowBlockForTextOutput implements Writable
	{
		public int numCols=0;
		public double[] container=null;
		public static enum Situation{START, NEWLINE, MIDDLE};
		Situation sit=Situation.START;
		//private StringBuffer buffer=new StringBuffer();
		private StringBuilder _buffer = new StringBuilder();
		
		String delim=",";
		boolean sparse=true;
		
		public RowBlockForTextOutput()
		{
		}
		
		public void set(MatrixBlock block, Situation s)
		{
			this.numCols=block.getNumColumns();
			container = DataConverter.convertToDoubleMatrix(block)[0];
			
			this.sit=s;
		}
		
		public void setSituation(Situation s)
		{
			this.sit=s;
		}
		
		public void setFormatParameters(String del, boolean sps)
		{
			delim=del;
			sparse=sps;
		}
		
		@Override
		public void readFields(DataInput arg0) throws IOException {
			throw new IOException("this is not supposed to be called!");
		}

		@Override
		public void write(DataOutput out) throws IOException {
			//long begin = System.nanoTime();
			_buffer.setLength(0);
			switch(sit)
			{
			case START:
				break;
			case NEWLINE:
				_buffer.append('\n');
				break;
			case MIDDLE:
				_buffer.append(delim);
				break;
			default:
				throw new RuntimeException("Unrecognized situation "+sit);	
			}
			
			if ( numCols > 0 ) {
				for(int j=0; j<numCols; j++)
				{
					double val = container[j];
					if( !sparse || val!=0 )
						_buffer.append(val);
					if( j < numCols-1 )
						_buffer.append(delim);
				}
			}
			
			ByteBuffer bytes = Text.encode(_buffer.toString());
			int length = bytes.limit();
		    out.write(bytes.array(), 0, length);
		}
		
	}
}
