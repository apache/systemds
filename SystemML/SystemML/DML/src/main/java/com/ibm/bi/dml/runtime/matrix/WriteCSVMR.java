/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVWriteInstruction;
import com.ibm.bi.dml.runtime.matrix.WriteCSVMR.RowBlockForTextOutput.Situation;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MapperBase;
import com.ibm.bi.dml.runtime.matrix.mapred.ReduceBase;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

public class WriteCSVMR 
{
    @SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		

	private static final Log LOG = LogFactory.getLog(WriteCSVMR.class.getName());
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, 
			long[] rlens, long[] clens, int[] brlens, int[] bclens, String csvWriteInstructions, int numReducers, int replication, 
			byte[] resultIndexes, String[] outputs) 
	throws Exception
	{
		//assert(inputs.length==outputs.length);
		JobConf job;
		job = new JobConf(WriteCSVMR.class);
		job.setJobName("WriteCSV-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, brlens, bclens, true, ConvertTarget.CSVWRITE);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		MRJobConfiguration.setCSVWriteInstructions(job, csvWriteInstructions);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		long maxRlen=0;
		for(long rlen: rlens)
			if(rlen>maxRlen)
				maxRlen=rlen;
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, maxRlen, numReducers);
		//job.setInt("mapred.tasktracker.map.tasks.maximum", 2);
		//job.setNumReduceTasks(2);
		
		byte[] resultDimsUnknown = new byte[resultIndexes.length];
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		OutputInfo[] outputInfos=new OutputInfo[outputs.length];
		HashMap<Byte, Integer> indexmap=new HashMap<Byte, Integer>();
		for(int i=0; i<stats.length; i++)
		{
			indexmap.put(resultIndexes[i], i);
			resultDimsUnknown[i] = (byte) 0;
			stats[i]=new MatrixCharacteristics();
			outputInfos[i]=OutputInfo.CSVOutputInfo;
		}
		CSVWriteInstruction[] ins = MRInstructionParser.parseCSVWriteInstructions(csvWriteInstructions);
		for(CSVWriteInstruction in: ins)
			stats[indexmap.get(in.output)].set(rlens[in.input], clens[in.input], -1, -1);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(stats);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  "", 
				"", csvWriteInstructions, resultIndexes);
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, 
				outputInfos, true, true);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(BreakMapper.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(RowBlock.class);
		
		//configure reducer
		job.setReducerClass(CombineReducer.class);
		job.setOutputKeyComparatorClass(TaggedFirstSecondIndexes.Comparator.class);
		job.setPartitionerClass(TaggedFirstSecondIndexes.FirstIndexRangePartitioner.class);
		//job.setOutputFormat(UnPaddedOutputFormat.class);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.REBLOCK, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
	
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		for(int i=0; i<resultIndexes.length; i++) {
			// number of non-zeros
			stats[i].nonZero=group.getCounter(Integer.toString(i));
			//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
		}
		
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
	
	public static class RowBlock implements Writable
	{
		public int numCols=0;
		public double[] container=null;
		
		@Override
		public void readFields(DataInput in) throws IOException {
			numCols=in.readInt();
			if(container==null || container.length<numCols)
				container=new double[numCols];
			for(int i=0; i<numCols; i++)
				container[i]=in.readDouble();
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(numCols);
			for(int i=0; i<numCols; i++)
				out.writeDouble(container[i]);
		}
		
		public String toString()
		{
			String str="";
			for(int i=0; i<numCols; i++)
				str+=container[i]+", ";
			return str;
		}
		
	}
	
	public static class RowBlockForTextOutput implements Writable
	{
		public int numCols=0;
		public double[] container=null;
		public static enum Situation{START, NEWLINE, MIDDLE};
		Situation sit=Situation.START;
		private StringBuffer buffer=new StringBuffer();
		String delim=",";
		boolean sparse=true;
		
		public RowBlockForTextOutput()
		{
		}
		
		public void set(RowBlock block, Situation s)
		{
			this.numCols=block.numCols;
			this.container=block.container;
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
			buffer.delete(0, buffer.length());
			switch(sit)
			{
			case START:
				break;
			case NEWLINE:
				buffer.append('\n');
				break;
			case MIDDLE:
				buffer.append(delim);
				break;
			default:
				throw new RuntimeException("Unrecognized situation "+sit);	
			}
			int i=0;
			for(; i<numCols-1; i++)
			{
				if( !sparse || container[i]!=0 )
					buffer.append(container[i]);
				buffer.append(delim);
			}
			if( !sparse || container[i]!=0 )
				buffer.append(container[i]);
			ByteBuffer bytes = Text.encode(buffer.toString());
		    int length = bytes.limit();
		    out.write(bytes.array(), 0, length);
		   // System.out.print(buffer.toString());
			//out.write(Text.encode(buffer.toString(), true).array());
		}
		public String toString()
		{
			String str="";
			for(int i=0; i<numCols; i++)
				str+=container[i]+", ";
			return str;
		}

		public long getNonZeros() {
			int n=0;
			for(int i=0; i<numCols; i++)
				if(container[i]!=0)
					n++;
			return n;
		}
	}
	
	static class BreakMapper extends MapperBase 
	implements Mapper<WritableComparable, Writable, TaggedFirstSecondIndexes, RowBlock>
	{
		HashMap<Byte, Vector<Byte>> inputOutputMap=new HashMap<Byte, Vector<Byte>>();
		TaggedFirstSecondIndexes outIndexes=new TaggedFirstSecondIndexes();
		@Override
		public void map(WritableComparable rawKey, Writable rawValue,
				OutputCollector<TaggedFirstSecondIndexes, RowBlock> out,
				Reporter reporter) throws IOException {
			long start=System.currentTimeMillis();
			
		//	System.out.println("read in Mapper: "+rawKey+": "+rawValue);
			
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
					Pair<MatrixIndexes, RowBlock> pair=inputConverter.next();
				//	System.out.println("convert to: "+pair);
					MatrixIndexes indexes=pair.getKey();
					
					RowBlock value=pair.getValue();
					
				//	System.out.println("after converter: "+indexes+" -- "+value);
					outIndexes.setIndexes(indexes.getRowIndex(), indexes.getColumnIndex());
					Vector<Byte> outputs=inputOutputMap.get(thisMatrix);
					for(byte output: outputs)
					{
						outIndexes.setTag(output);
						out.collect(outIndexes, value);
						//LOG.info("Mapper output: "+outIndexes+", "+value+", tag: "+output);
					}
				}
			}
			reporter.incrCounter(Counters.MAP_TIME, System.currentTimeMillis()-start);
			
		}
		
		public void configure(JobConf job)
		{
			super.configure(job);
			try {
				CSVWriteInstruction[] ins = MRJobConfiguration.getCSVWriteInstructions(job);
				for(CSVWriteInstruction in: ins)
				{
					Vector<Byte> outputs=inputOutputMap.get(in.input);
					if(outputs==null)
					{
						outputs=new Vector<Byte>();
						inputOutputMap.put(in.input, outputs);
					}
					outputs.add(in.output);
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
		
		@Override
		protected void specialOperationsForActualMap(int index,
				OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException {
			// do nothing
		}
	}
	
	static class CombineReducer extends ReduceBase 
	implements Reducer<TaggedFirstSecondIndexes, RowBlock, NullWritable, RowBlockForTextOutput>
	{
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
		
		@Override
		public void reduce(TaggedFirstSecondIndexes inkey,
				Iterator<RowBlock> inValue,
				OutputCollector<NullWritable, RowBlockForTextOutput> out, Reporter reporter)
				throws IOException {
		
			if(firsttime)
			{
				cachedReporter=reporter;
				firsttime=false;
			}
			
			byte tag=inkey.getTag();
			zeroBlock.setFormatParameters(delims[tag], sparses[tag]);
			outValue.setFormatParameters(delims[tag], sparses[tag]);
			//LOG.info("~~~~~ Reducer read: "+inkey+", "+inValue);
			Situation sit=Situation.MIDDLE;
			if(rowIndexes[tag]==minRowIndexes[tag])
				sit=Situation.START;
			else if(rowIndexes[tag]!=inkey.getFirstIndex())
				sit=Situation.NEWLINE;
			
			//check whether need to fill in missing values in previous rows
			if(sit==Situation.NEWLINE)
			{
				//if the previous row has not finished
				//LOG.info("~~~the previous row has not finished");
				addEndingMissingValues(tag, reporter);
			}
			if(sit==Situation.NEWLINE||sit==Situation.START)
			{	
				//if a row is completely missing
				//LOG.info("~~~~a row is completely missing");
				sit=addMissingRows(tag, inkey.getFirstIndex(), sit, reporter);
			}
			//add missing value at the beginning of this row
			//LOG.info("~~~~add missing value at the beginning of this row");
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
				//LOG.info("~~ in loop output ");
				outValue.set(inValue.next(), sit);
				collectFinalMultipleOutputs.directOutput(nullKey, outValue, tagToResultIndex[tag], reporter);
				resultsNonZeros[tagToResultIndex[tag]]+=outValue.getNonZeros();
				sit=RowBlockForTextOutput.Situation.MIDDLE;
			}
			rowIndexes[tag]=inkey.getFirstIndex();
		}
		
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
				//LOG.info("## rowIndex: "+rowIndexes[ri]+", maxRowIndexes: "+maxRowIndexes[ri]);
			}
			
			zeroBlock.container=new double[maxCol];
		}
		public void close() throws IOException
		{
			for(byte tag:resultIndexes)
			{
				//if the previous row has not finished
				addEndingMissingValues(tag, cachedReporter);
				
				//if a row is completely missing
				addMissingRows(tag, maxRowIndexes[tag]+1, Situation.NEWLINE, cachedReporter);
			}
			super.close();
		}
	}
	
	public static void main(String[] args) throws Exception {
		ConfigurationManager.setConfig(new DMLConfig("SystemML-config.xml"));
		String[] inputs = {"data/test1.csv", "data/test2.csv"};
		InputInfo[] inputInfos = {InputInfo.CSVInputInfo, InputInfo.CSVInputInfo};
		String[] outputs = {"data/A.tmp", "data/B.tmp"};
		OutputInfo[] outputInfos = {OutputInfo.BinaryBlockOutputInfo, OutputInfo.BinaryBlockOutputInfo};
		int[] brlens = { 1000, 1000};
		int[] bclens = { 1000, 1000};
		
		String ins1= "MR" + Instruction.OPERAND_DELIM 
		+ "csvrblk" + Instruction.OPERAND_DELIM 
		+ 0 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 2 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ brlens[0] + Instruction.OPERAND_DELIM
		+ bclens[0] + Instruction.OPERAND_DELIM
		+ "," + Instruction.OPERAND_DELIM
		+ false+ Instruction.OPERAND_DELIM
		+ true+ Instruction.OPERAND_DELIM
		+100.0;
		
		String ins2= "MR" + Instruction.OPERAND_DELIM 
		+ "csvrblk" + Instruction.OPERAND_DELIM 
		+ 1 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 3 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ brlens[1] + Instruction.OPERAND_DELIM
		+ bclens[1] + Instruction.OPERAND_DELIM
		+ "," + Instruction.OPERAND_DELIM
		+ false+ Instruction.OPERAND_DELIM
		+ true+ Instruction.OPERAND_DELIM
		+0.0;
		
		JobReturn ret=CSVReblockMR.runJob(null, inputs, inputInfos, new long[]{-1, -1}, new long[]{-1, -1}, brlens, bclens, ins1+ Instruction.INSTRUCTION_DELIM +ins2, null, 2, 1, new byte[]{2,3}, outputs, 
				outputInfos);
		System.out.println("END of CSVReblock");
		
		InputInfo[] inputInfos2 = {InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo};
		long[] rlens= new long[]{((MatrixDimensionsMetaData)(ret.metadata[0])).matchar.numRows, 
				((MatrixDimensionsMetaData)(ret.metadata[1])).matchar.numRows};
		long[] clens= new long[]{((MatrixDimensionsMetaData)(ret.metadata[0])).matchar.numColumns, 
				((MatrixDimensionsMetaData)(ret.metadata[1])).matchar.numColumns};
		String ins3= "MR" + Instruction.OPERAND_DELIM 
		+ "csvwrite" + Instruction.OPERAND_DELIM 
		+ 0 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 2 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ true + Instruction.OPERAND_DELIM
		+ "@" + Instruction.OPERAND_DELIM
		+ false;
		
		String ins4= "MR" + Instruction.OPERAND_DELIM 
		+ "csvwrite" + Instruction.OPERAND_DELIM 
		+ 1 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 3 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ false + Instruction.OPERAND_DELIM
		+ "%" + Instruction.OPERAND_DELIM
		+ true;
		String[] outputs2 = {"data/A.out", "data/B.out"};
		WriteCSVMR.runJob(null, outputs, inputInfos2, rlens, clens, brlens, bclens, ins3+Instruction.INSTRUCTION_DELIM+ins4, 2, 1, new byte[]{2,3}, outputs2);
	}
	
}
