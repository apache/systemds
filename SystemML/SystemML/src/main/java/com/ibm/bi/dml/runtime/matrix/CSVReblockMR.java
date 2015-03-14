/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashSet;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVAssignRowIDMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVAssignRowIDReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockMapper;
import com.ibm.bi.dml.runtime.matrix.mapred.CSVReblockReducer;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.MatrixChar_N_ReducerGroups;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class CSVReblockMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String NUM_ROWS_IN_MATRIX="num.rows.in.matrix.";
	public static final String NUM_COLS_IN_MATRIX="num.cols.in.matrix.";
	public static final String ROWID_FILE_NAME="rowid.file.name";
	public static final String SMALLEST_FILE_NAME_PER_INPUT="smallest.file.name.per.input";
	
	private CSVReblockMR() {
		//prevent instantiation via private constructor
	}
	
	public static final PathFilter hiddenFileFilter = new PathFilter(){
	      public boolean accept(Path p){
	        String name = p.getName(); 
	        return !name.startsWith("_") && !name.startsWith("."); 
	      }
	    }; 
	
	public static class OffsetCount implements WritableComparable
	{
		public String filename;
		public long fileOffset;
		public long count;
		
		public OffsetCount()
		{
			filename="";
			fileOffset=0;
			count=0;
		}
		
		public OffsetCount(String fname, long off, long cnt)
		{
			filename=fname;
			fileOffset=off;
			count=cnt;
		}
		
		public OffsetCount(OffsetCount that)
		{
			this.filename=that.filename;
			this.fileOffset=that.fileOffset;
			this.count=that.count;
		}
		
		@Override
		public void readFields(DataInput in) throws IOException {
			filename=in.readLine();
			fileOffset=in.readLong();
			count=in.readLong();
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeBytes(filename+'\n');
			out.writeLong(fileOffset);
			out.writeLong(count);
		}
		
		public String toString()
		{
			return filename+", "+fileOffset+", "+count;
		}

		public int compareTo(OffsetCount that) {
			int ret=this.filename.compareTo(that.filename);
			if(ret!=0)
				return ret;
			else if(this.fileOffset<that.fileOffset)
				return -1;
			else if(this.fileOffset>that.fileOffset)
				return 1;
			else return 0;
		}
		
		@Override
		public int compareTo(Object o) {
			if( !(o instanceof OffsetCount) )
				return -1;
			return compareTo((OffsetCount)o);
		}
		
		@Override
		public boolean equals(Object o) {
			if( !(o instanceof OffsetCount) )
				return false;
			OffsetCount that = (OffsetCount)o;
			return (filename.equals(that.filename) && fileOffset==that.fileOffset);
		}
		
		@Override
		public int hashCode() {
			throw new RuntimeException("hashCode() should never be called on instances of this class.");
		}
	}
	
	public static class BlockRow implements Writable
	{
		public int indexInBlock = 0;
		public MatrixBlock data = null;
		
		@Override
		public void readFields(DataInput in) throws IOException {
			indexInBlock=in.readInt();
			if( data == null )
				data = new MatrixBlock();
			data.readFields(in);
		}
		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(indexInBlock);
			data.write(out);
		}
	}
	
	public static class AssignRowIDMRReturn
	{
		public Path counterFile=null;
		public long[] rlens=null;
		public long[] clens=null;
		
		public String toString()
		{
			StringBuilder sb = new StringBuilder();
			sb.append(counterFile.toString());
			sb.append("\n");
			for(long rlen: rlens) {
				sb.append(rlen);
				sb.append(", ");
			}
			sb.append("\n");
			for(long clen: clens) {
				sb.append(clen);
				sb.append(", ");
			}
			
			return sb.toString();
		}
	}
	
	public static JobReturn runJob(MRJobInstruction inst, String[] inputs, InputInfo[] inputInfos, long[] rlens, long[] clens, 
			int[] brlens, int[] bclens, String reblockInstructions, 
			String otherInstructionsInReducer, int numReducers, int replication, byte[] resultIndexes, 
			String[] outputs, OutputInfo[] outputInfos) throws Exception 
	{
		String[] smallestFiles=new String[inputs.length];
		JobConf job=new JobConf();
		for(int i=0; i<inputs.length; i++)
		{
			Path p=new Path(inputs[i]);
			FileSystem fs = p.getFileSystem(job);
			if(!fs.isDirectory(p))
				smallestFiles[i]=p.makeQualified(fs).toString();
			else
			{
				FileStatus[] stats=fs.listStatus(p, hiddenFileFilter);
				if(stats.length==0)
					smallestFiles[i]="";
				else
				{
					smallestFiles[i]=stats[0].getPath().toString();
					for(int j=1; j<stats.length; j++)
					{
						String f=stats[j].getPath().toString();
						if(f.compareTo(smallestFiles[i])<0)
							smallestFiles[i]=f;
					}
				}
			}
		}
		
		AssignRowIDMRReturn ret1 = CSVReblockMR.runAssignRowIDMRJob(inputs, inputInfos, brlens, bclens, reblockInstructions, 
				replication, smallestFiles);
		for(int i=0; i<rlens.length; i++)
			if( (rlens[i]>0 && rlens[i]!=ret1.rlens[i]) || (clens[i]>0 && clens[i]!=ret1.clens[i]) )
				throw new RuntimeException("Dimension doesn't mach for input matrix "+i+", expected ("+rlens[i]+", "+clens[i]+") but real ("+ret1.rlens[i]+", "+ret1.clens[i]+")");
		JobReturn ret= CSVReblockMR.runCSVReblockJob(null, inputs, inputInfos, ret1.rlens, ret1.clens, brlens, bclens, reblockInstructions, 
				otherInstructionsInReducer, numReducers, replication, resultIndexes, outputs, outputInfos, ret1.counterFile, smallestFiles);
		return ret;
	}
	
	private static AssignRowIDMRReturn runAssignRowIDMRJob(String[] inputs, InputInfo[] inputInfos, int[] brlens, int[] bclens, 
			String reblockInstructions, int replication, String[] smallestFiles) 
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
				brlens, bclens, false, ConvertTarget.CELL);
		
		job.setStrings(SMALLEST_FILE_NAME_PER_INPUT, smallestFiles);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setCSVReblockInstructions(job, reblockInstructions);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up the number of reducers
		job.setNumReduceTasks(1);
		
		// Print the complete instruction
		//if (LOG.isTraceEnabled())
			//inst.printCompelteMRJobInstruction();
		
		// configure mapper and the mapper output key value pairs
		job.setMapperClass(CSVAssignRowIDMapper.class);
		job.setMapOutputKeyClass(ByteWritable.class);
		job.setMapOutputValueClass(OffsetCount.class);
		
		//configure reducer
		job.setReducerClass(CSVAssignRowIDReducer.class);
		
		/*
		 	job.setBoolean("adaptivemr.map.enable", true);
			job.setInt("adaptivemr.map.waves", 1);
		 */
		
		//turn off adaptivemr
		job.setBoolean("adaptivemr.map.enable", false);
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		
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
			String[] outputs, OutputInfo[] outputInfos, Path counterFile, String[] smallestFiles) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(ReblockMR.class);
		job.setJobName("CSV-Reblock-MR");
		
		byte[] realIndexes=new byte[inputs.length];
		for(byte b=0; b<realIndexes.length; b++)
			realIndexes[b]=b;
		
		//set up the input files and their format information
		MRJobConfiguration.setUpMultipleInputs(job, realIndexes, inputs, inputInfos, brlens, bclens, false, ConvertTarget.CELL);
		
		job.setStrings(SMALLEST_FILE_NAME_PER_INPUT, smallestFiles);
		
		//set up the dimensions of input matrices
		MRJobConfiguration.setMatricesDimensions(job, realIndexes, rlens, clens);
		
		//set up the block size
		MRJobConfiguration.setBlocksSizes(job, realIndexes, brlens, bclens);
		
		//set up the aggregate instructions that will happen in the combiner and reducer
		MRJobConfiguration.setCSVReblockInstructions(job, reblockInstructions);
		
		//set up the instructions that will happen in the reducer, after the aggregation instrucions
		MRJobConfiguration.setInstructionsInReducer(job, otherInstructionsInReducer);
		
		//set up the replication factor for the results
		job.setInt("dfs.replication", replication);
		
		//set up preferred custom serialization framework for binary block format
		if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
			MRJobConfiguration.addBinaryBlockSerializationFramework( job );
		
		//set up what matrices are needed to pass from the mapper to reducer
		HashSet<Byte> mapoutputIndexes=MRJobConfiguration.setUpOutputIndexesForMapper(job, realIndexes,  null, 
				reblockInstructions, null, otherInstructionsInReducer, resultIndexes);
		
		MatrixChar_N_ReducerGroups ret=MRJobConfiguration.computeMatrixCharacteristics(job, realIndexes, 
				null, reblockInstructions, null, null, otherInstructionsInReducer, 
				resultIndexes, mapoutputIndexes, false);
		
		MatrixCharacteristics[] stats=ret.stats;
		
		//set up the number of reducers
		int numRed = WriteCSVMR.determineNumReducers(rlens, clens, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS), ret.numReducerGroups);
		job.setNumReduceTasks( numRed );
		
		// Print the complete instruction
		//if (LOG.isTraceEnabled())
		//	inst.printCompelteMRJobInstruction(stats);
		
		// Update resultDimsUnknown based on computed "stats"
		byte[] resultDimsUnknown = new byte[resultIndexes.length];
		for ( int i=0; i < resultIndexes.length; i++ ) { 
			if ( stats[i].getRows() == -1 || stats[i].getCols() == -1 ) {
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
	
		//turn off adaptivemr
		job.setBoolean("adaptivemr.map.enable", false);
		
		
		// By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
		
		// at this point, both reblock_binary and reblock_text are similar
		MatrixCharacteristics[] inputStats = new MatrixCharacteristics[inputs.length];
		for ( int i=0; i < inputs.length; i++ ) {
			inputStats[i] = new MatrixCharacteristics(rlens[i], clens[i], brlens[i], bclens[i]);
		}
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		Path cachefile=new Path(counterFile, "part-00000");
		DistributedCache.addCacheFile(cachefile.toUri(), job);
		DistributedCache.createSymlink(job);
		job.set(ROWID_FILE_NAME, cachefile.toString());
		
		RunningJob runjob=JobClient.runJob(job);
		
		MapReduceTool.deleteFileIfExistOnHDFS(counterFile, job);
		
		/* Process different counters */
		
		Group group=runjob.getCounters().getGroup(MRJobConfiguration.NUM_NONZERO_CELLS);
		for(int i=0; i<resultIndexes.length; i++) {
			// number of non-zeros
			stats[i].setNonZeros(group.getCounter(Integer.toString(i)));
			//	System.out.println("result #"+resultIndexes[i]+" ===>\n"+stats[i]);
		}
		return new JobReturn(stats, outputInfos, runjob.isSuccessful());
	}
	
}
