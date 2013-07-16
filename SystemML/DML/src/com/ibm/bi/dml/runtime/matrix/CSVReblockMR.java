package com.ibm.bi.dml.runtime.matrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CSVReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MapperBase;
import com.ibm.bi.dml.runtime.matrix.mapred.ReduceBase;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


public class CSVReblockMR {
	private static final Log LOG = LogFactory.getLog(CSVReblockMR.class.getName());
	
	public static final String NUM_ROWS_IN_MATRIX="num.rows.in.matrix.";
	public static final String NUM_COLS_IN_MATRIX="num.cols.in.matrix.";
	public static final String ROWID_FILE_NAME="rowid.file.name";
	
	static class OffsetCount implements Writable
	{
		public OffsetCount()
		{
			fileOffset=0;
			count=0;
		}
		
		public OffsetCount(long off, long cnt)
		{
			fileOffset=off;
			count=cnt;
		}
		public long fileOffset;
		public long count;
		@Override
		public void readFields(DataInput in) throws IOException {
			fileOffset=in.readLong();
			count=in.readLong();
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeLong(fileOffset);
			out.writeLong(count);
		}
		
		public String toString()
		{
			return fileOffset+", "+count;
		}
		
	}
	
	static class AssignRowIDMapper extends MapReduceBase implements Mapper<LongWritable, Text, ByteWritable, OffsetCount>
	{
		ByteWritable outKey=new ByteWritable();
		long fileOffset=0;
		long num=0;
		boolean first=true;
		OutputCollector<ByteWritable, OffsetCount> outCache=null;
		String delim=" ";
		boolean ignoreFirstLine=false;
		boolean realFirstLine=false;
		public void configure(JobConf job)
		{
			byte thisIndex;
			try {
				//it doesn't make sense to have repeated file names in the input, since this is for reblock
				thisIndex=MRJobConfiguration.getInputMatrixIndexesInMapper(job).get(0);
				outKey.set(thisIndex);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
			
			try {
				CSVReblockInstruction[] reblockInstructions = MRJobConfiguration.getCSVReblockInstructions(job);
				for(CSVReblockInstruction ins: reblockInstructions)
				{
					if(ins.input==thisIndex)
					{
						delim=Character.toString(ins.delim);
						ignoreFirstLine=ins.ignoreFirstLine;
						break;
					}
				}
			} catch (DMLUnsupportedOperationException e) {
				throw new RuntimeException(e);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
		
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<ByteWritable, OffsetCount> out, Reporter report)
				throws IOException {
			if(first)
			{
				first=false;
				fileOffset=key.get();
				outCache=out;
			}
			if(key.get()==0)//getting the number of colums
			{
				if(!ignoreFirstLine)
				{
					report.incrCounter(NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim).length);
				//	System.out.println(value.toString().split(delim).length);
					num++;
				}
				else
					realFirstLine=true;
			}else
			{
				if(realFirstLine)
				{
					report.incrCounter(NUM_COLS_IN_MATRIX, outKey.toString(), value.toString().split(delim).length);
					realFirstLine=false;
				}
				num++;
			}
			//System.out.println(key+": "+value);
			//System.out.println("num: "+num);
		}
		
		public void close() throws IOException
		{
			outCache.collect(outKey, new OffsetCount(fileOffset, num));
		}
	}
	
	static class AssignRowIDReducer extends MapReduceBase implements Reducer<ByteWritable, OffsetCount, ByteWritable, OffsetCount>
	{
		@Override
		public void reduce(ByteWritable key, Iterator<OffsetCount> values,
				OutputCollector<ByteWritable, OffsetCount> out, Reporter report)
				throws IOException {
			long lineOffset=0;
			while(values.hasNext())
			{
				OffsetCount oc=values.next();
				long count=oc.count;
				oc.count=lineOffset;
				out.collect(key, oc);
				//System.out.println(key+": "+oc);
				lineOffset+=count;
			}
			report.incrCounter(NUM_ROWS_IN_MATRIX, key.toString(), lineOffset);
		}
	}
	
	static class BlockRow implements Writable
	{
		public int indexInBlock=0;
		public int numCols=0;
		public double[] container=null;
		
		@Override
		public void readFields(DataInput in) throws IOException {
			indexInBlock=in.readInt();
			numCols=in.readInt();
			if(container==null || container.length<numCols)
				container=new double[numCols];
			for(int i=0; i<numCols; i++)
				container[i]=in.readDouble();
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(indexInBlock);
			out.writeInt(numCols);
			for(int i=0; i<numCols; i++)
				out.writeDouble(container[i]);
		}
		
		public String toString()
		{
			String str="index: "+indexInBlock+"\n";
			for(int i=0; i<numCols; i++)
				str+=container[i]+", ";
			return str;
		}
		
	}
	
	static class CSVReblockMapper extends MapperBase implements Mapper<LongWritable, Text, TaggedFirstSecondIndexes, BlockRow>
	{
		long rowOffset=0;
		boolean first=true;
		long num=0;
		HashMap<Long, Long> offsetMap=new HashMap<Long, Long>();
		TaggedFirstSecondIndexes outIndexes=new TaggedFirstSecondIndexes();
		BlockRow row;
		String delim=" ";
		boolean ignoreFirstLine=false;
		double missingValue=0;
		
		public void configure(JobConf job)
		{
			super.configure(job);
			//get the number colums per block
			row=new BlockRow();
			int maxBclen=0;
			for(int i=0; i<bclens.length; i++)
				if(maxBclen<bclens[i])
					maxBclen=bclens[i];
			row.container=new double[maxBclen];		
			
			//load the offset mapping
			byte matrixIndex=representativeMatrixes.get(0);
			try {
				Path[] paths=DistributedCache.getLocalCacheFiles(job);
				FileSystem fs = FileSystem.get(job);
				ByteWritable key=new ByteWritable();
				OffsetCount value=new OffsetCount();
				Path p=new Path(job.get(ROWID_FILE_NAME));
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
				try {
					while (reader.next(key, value)) {
						if(key.get()==matrixIndex)
							offsetMap.put(value.fileOffset, value.count);
					}
				} catch (Exception e) {
					throw new RuntimeException(e);
				} 
				
				reader.close();
				/*for(Path p: paths)
				{
					SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
					try {
						while (reader.next(key, value)) {
							if(key.get()==matrixIndex)
								offsetMap.put(value.fileOffset, value.count);
						}
					} catch (Exception e) {
						throw new RuntimeException(e);
					} 
					
					reader.close();
				}*/
				
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
			CSVReblockInstruction ins=csv_reblock_instructions.get(0).get(0);
			delim=Character.toString(ins.delim);
			ignoreFirstLine=ins.ignoreFirstLine;
			missingValue=ins.missingValue;
		}
		
		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<TaggedFirstSecondIndexes, BlockRow> out, Reporter reporter)
				throws IOException {
			if(first)
			{
				rowOffset=offsetMap.get(key.get());
				first=false;
			}
			
			if(key.get()==0 && ignoreFirstLine)
				return;
			
			String[] cells=value.toString().split(delim);
			
			for(int i=0; i<representativeMatrixes.size(); i++)
			{
				int start=0;
				row.numCols=bclens[i];
				outIndexes.setTag(csv_reblock_instructions.get(i).get(0).output);
				long rowIndex=UtilFunctions.blockIndexCalculation(rowOffset+num+1, brlens[i]);
				row.indexInBlock=UtilFunctions.cellInBlockCalculation(rowOffset+num+1, brlens[i]);
				
				long col=0;
				for(; col<cells.length/bclens[i]; col++)
				{
					outIndexes.setIndexes(rowIndex, col+1);
					for(int k=0;k<bclens[i]; k++)
					{
						if(cells[k+start].isEmpty())
							row.container[k]=missingValue;
						else
							row.container[k]= Double.parseDouble(cells[k+start]);
					}
					out.collect(outIndexes, row);
					//System.out.println("mapper: "+outIndexes+", "+row);
					start+=bclens[i];
				}
				outIndexes.setIndexes(rowIndex, col+1);
				int lastBclen=cells.length%bclens[i];
				if(lastBclen!=0)
				{
					row.numCols=lastBclen+1;
					for(int k=0;k<lastBclen; k++)
						row.container[k]= Double.parseDouble(cells[k+start]);
					out.collect(outIndexes, row);
					//System.out.println("mapper: "+outIndexes+", "+row);
				}
			}
			
			num++;
		}

		@Override
		protected void specialOperationsForActualMap(int index,
				OutputCollector<Writable, Writable> out, Reporter reporter)
				throws IOException {
		}

		
	}
	
	static class CSVReblockReducer extends ReduceBase 
	implements Reducer<TaggedFirstSecondIndexes, BlockRow, MatrixIndexes, MatrixBlock>
	{
		@Override
		public void reduce(TaggedFirstSecondIndexes key, Iterator<BlockRow> values,
				OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter)
				throws IOException {
			
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
		
		protected void processCSVReblock(TaggedFirstSecondIndexes indexes, Iterator<BlockRow> values, 
				HashMap<Byte, MatrixCharacteristics> dimensions)
		{
//			System.out.println("in Reducer: "+indexes+": "+partial);
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
				
				((MatrixBlock) block.getValue()).copyRowArrayToDense((int)row.indexInBlock, row.container, 0);
			}
			((MatrixBlock) block.getValue()).recomputeNonZeros();
			System.out.println("block+\n"+block);
		}
		
		public void configure(JobConf job) {
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
	
	public static class AssignRowIDMRReturn
	{
		public Path counterFile=null;
		public long[] rlens=null;
		public long[] clens=null;
		public String toString()
		{
			String str=counterFile.toString()+"\n";
			for(long rlen: rlens)
				str+=rlen+", ";
			str+="\n";
			for(long clen: clens)
				str+=clen+", ";
			return str;
		}
	}
	
	private static AssignRowIDMRReturn runAssignRowIDMRJob(String[] inputs, InputInfo[] inputInfos, int[] brlens, int[] bclens, 
			String reblockInstructions, int replication) 
	throws Exception
	{
		AssignRowIDMRReturn ret=new AssignRowIDMRReturn();
		JobConf job;
		job = new JobConf(CSVReblockMR.class);
		job.setJobName("Assign-RowID-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, 
				false, brlens, bclens, false, false);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setReblockInstructions(job, reblockInstructions);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up the number of reducers
		job.setNumReduceTasks(1);
		
		// Print the complete instruction
		//if (LOG.isTraceEnabled())
			//inst.printCompelteMRJobInstruction();
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(AssignRowIDMapper.class);
		job.setMapOutputKeyClass(ByteWritable.class);
		job.setMapOutputValueClass(OffsetCount.class);
		
		//configure reducer
		job.setReducerClass(AssignRowIDReducer.class);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, ExecMode.CLUSTER);
		
		//set up the output file
		ret.counterFile=new Path(MRJobConfiguration.constructTempOutputFilename());
		job.setOutputFormat(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, ret.counterFile);
		job.setOutputKeyClass(ByteWritable.class);
		job.setOutputValueClass(OffsetCount.class);
		RunningJob runjob=JobClient.runJob(job);
		
		/* Process different counters */
		
		Group rgroup=runjob.getCounters().getGroup(NUM_ROWS_IN_MATRIX);
		Group cgroup=runjob.getCounters().getGroup(NUM_COLS_IN_MATRIX);
		ret.rlens=new long[inputs.length];
		ret.clens=new long[inputs.length];
		for(int i=0; i<inputs.length; i++) {
			// number of non-zeros
			ret.rlens[i]=rgroup.getCounter(Integer.toString(i));
			ret.clens[i]=cgroup.getCounter(Integer.toString(i));
		}
		return ret;
	}
	
	private static JobReturn runCSVReblockJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String reblockInstructions, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, 
			String[] outputs, OutputInfo[] outputInfos, Path counterFile) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(ReblockMR.class);
		job.setJobName("CVS-Reblock-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, false, brlens, bclens);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setReblockInstructions(job, reblockInstructions);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  null, 
				reblockInstructions, null, otherInstructionsInReducer, resultIndexes);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				null, reblockInstructions, null, null, otherInstructionsInReducer, 
				resultIndexes, mapoutputIndexes, false);
		
		MatrixCharacteristics[] stats=ret.stats;
		
		//set up the number of reducers
		MRJobConfiguration.setNumReducers(job, ret.numReducerGroups, numReducers);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompelteMRJobInstruction(stats);
		
		// Update resultDimsUnknown based on computed "stats"
		byte[] resultDimsUnknown = new byte[resultIndexes.length];
		for ( int i=0; i < resultIndexes.length; i++ ) { 
			if ( stats[i].numRows == -1 || stats[i].numColumns == -1 ) {
				resultDimsUnknown[i] = (byte) 1;
			}
			else {
				resultDimsUnknown[i] = (byte) 0;
			}
		}
		
		//set up the multiple output files, and their format information
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true, true);
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(CSVReblockMapper.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(BlockRow.class);
		
		//configure reducer
		job.setReducerClass(CSVReblockReducer.class);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		// at this point, both reblock_binary and reblock_text are similar
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		ExecMode mode = RunMRJobs.getExecMode(JobType.REBLOCK_BINARY, inputStats); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		Path cachefile=new Path(counterFile, "part-00000");
		DistributedCache.addCacheFile(cachefile.toUri(), job);
		DistributedCache.createSymlink(job);
		job.set(ROWID_FILE_NAME, cachefile.toString());
		
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
	
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String reblockInstructions, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, 
			String[] outputs, OutputInfo[] outputInfos) throws Exception 
	{
		AssignRowIDMRReturn ret1 = CSVReblockMR.runAssignRowIDMRJob(inputs, inputInfos, brlens, bclens, reblockInstructions, replication);
		return CSVReblockMR.runCSVReblockJob(null, inputs, inputInfos, ret1.rlens, ret1.clens, brlens, bclens, reblockInstructions, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, outputs, outputInfos, ret1.counterFile);
	}
	
	public static void main(String[] args) throws Exception {
		ConfigurationManager.setConfig(new DMLConfig("SystemML-config.xml"));
		String[] inputs = {"data/A.csv", "data/B.csv"};
		InputInfo[] inputInfos = {InputInfo.CSVInputInfo, InputInfo.CSVInputInfo};
		String[] outputs = {"data/A.out", "data/B.out"};
		OutputInfo[] outputInfos = {OutputInfo.TextCellOutputInfo, OutputInfo.TextCellOutputInfo};
		int[] brlens = { 2, 2};
		int[] bclens = { 2, 2};
		
		String ins1= "MR" + Instruction.OPERAND_DELIM 
		+ "rblkcsv" + Instruction.OPERAND_DELIM 
		+ 0 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 2 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ brlens[0] + Instruction.OPERAND_DELIM
		+ bclens[0] + Instruction.OPERAND_DELIM
		+ Byte.toString((byte)'%') + Instruction.OPERAND_DELIM
		+ true+ Instruction.OPERAND_DELIM
		+100.0;
		
		String ins2= "MR" + Instruction.OPERAND_DELIM 
		+ "rblkcsv" + Instruction.OPERAND_DELIM 
		+ 1 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM  
		+ 3 + Instruction.DATATYPE_PREFIX + DataType.MATRIX + Instruction.VALUETYPE_PREFIX + ValueType.DOUBLE + Instruction.OPERAND_DELIM
		+ brlens[1] + Instruction.OPERAND_DELIM
		+ bclens[1] + Instruction.OPERAND_DELIM
		+ Byte.toString((byte)' ') + Instruction.OPERAND_DELIM
		+ false+ Instruction.OPERAND_DELIM
		+0.0;
		
		CSVReblockMR.runJob(null, inputs, inputInfos, null, null, brlens, bclens, ins1+","+ins2, null, 2, 1, new byte[]{2,3}, outputs, 
				outputInfos);
	}
}
