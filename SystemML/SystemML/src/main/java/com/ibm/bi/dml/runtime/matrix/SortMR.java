/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration.ConvertTarget;
import com.ibm.bi.dml.runtime.matrix.sort.CompactInputFormat;
import com.ibm.bi.dml.runtime.matrix.sort.CompactOutputFormat;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortComparable;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortComparableDesc;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortMapper;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortReducer;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortStitchupReducer;
import com.ibm.bi.dml.runtime.matrix.sort.SamplingSortMRInputFormat;
import com.ibm.bi.dml.runtime.matrix.sort.IndexSortStitchupMapper;
import com.ibm.bi.dml.runtime.matrix.sort.ValueSortMapper;
import com.ibm.bi.dml.runtime.matrix.sort.ValueSortReducer;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


/**
 * TODO fix issues sortindex mappers
 */
public class SortMR 
{

    @SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    private static final Log LOG = LogFactory.getLog(SortMR.class.getName());
    
    public static final String NUM_VALUES_PREFIX="num.values.in";
    public static final String COMBINE_INSTRUCTION = "combine.instruction";
    public static final String SORT_INSTRUCTION = "sort.instruction";
    public static final String VALUE_IS_WEIGHT="value.is.weight";
    public static final String SORT_INDEXES_OFFSETS = "sort.indexes.offsets";
    public static final String SORT_DECREASING = "sort.decreasing";
    
    
  	private SortMR() {
		//prevent instantiation via private constructor
	}
  
  
  /**
   * A partitioner that splits text keys into roughly equal partitions
   * in a global sorted order.
   */
  @SuppressWarnings({ "unchecked", "rawtypes" })
  private static class TotalOrderPartitioner<K extends WritableComparable, V extends Writable> 
                      implements Partitioner<K, V>
  { 
	private ArrayList<WritableComparable> splitPoints;
    private Class<? extends WritableComparable> keyClass;
    private Class<? extends Writable> valueClass;
    
    /**
     * Read the cut points from the given sequence file.
     * @param fs the file system
     * @param p the path to read
     * @param job the job config
     * @return the strings to split the partitions on
     * @throws IOException
     * @throws IllegalAccessException 
     * @throws InstantiationException 
     */
    private ArrayList<WritableComparable> readPartitions(FileSystem fs, Path p, JobConf job) 
    	throws IOException 
    {
    	SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
    	ArrayList<WritableComparable> parts = new ArrayList<WritableComparable>();
    	try 
    	{
			//WritableComparable key = keyClass.newInstance();
    		DoubleWritable key = new DoubleWritable();
			NullWritable value = NullWritable.get();
			while (reader.next(key, value)) {
				parts.add(key);
				//key=keyClass.newInstance();
				key = new DoubleWritable();
			}
		} 
    	catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		reader.close();
		return parts;
    }

    public void configure(JobConf job) {
      try {
    	  keyClass=(Class<? extends WritableComparable>) job.getMapOutputKeyClass();
          valueClass=(Class<? extends Writable>) job.getMapOutputValueClass();
          FileSystem fs = FileSystem.get(job);
          Path partFile = new Path(MRJobConfiguration.getSortPartitionFilename(job)); 
          splitPoints = readPartitions(fs, partFile, job);
        
      } 
      catch (IOException ie) {
        throw new IllegalArgumentException("can't read paritions file", ie);
      }
    }

    public TotalOrderPartitioner() {
    	
    }

    public int getPartition(K key, V value, int numPartitions) {
      return findPartition(key)%numPartitions;
    }

	private int findPartition(K key) {
		int i=0;
		for( ; i<splitPoints.size(); i++)
		{
			//always ascending (support for ascending/descending) is
			//controlled via IndexSortComparable/IndexSortComparableDesc
			if(key.compareTo(splitPoints.get(i))<=0)
				return i;
			
		}
		return i;
	}
    
  }
  
	public static JobReturn runJob(MRJobInstruction inst, String input, InputInfo inputInfo, long rlen, long clen, 
			int brlen, int bclen, String combineInst, String sortInst, int numReducers, 
			int replication, String output, OutputInfo outputInfo, boolean valueIsWeight) 
	  throws Exception 
	  {
	    boolean sortIndexes = getSortInstructionType(sortInst)==SortKeys.OperationTypes.Indexes;
	    String tmpOutput = sortIndexes ? MRJobConfiguration.constructTempOutputFilename() : output;
	    
	    JobConf job = new JobConf(SortMR.class);
	    job.setJobName("SortMR");
	    
	    //setup partition file
	    String pfname = MRJobConfiguration.setUpSortPartitionFilename(job);
	    Path partitionFile = new Path( pfname ); 
	    URI partitionUri = new URI( partitionFile.toString() ); 
	   
	    //setup input/output paths
	    Path inputDir = new Path(input);
	    inputDir = inputDir.makeQualified(inputDir.getFileSystem(job));
	    SamplingSortMRInputFormat.setInputPaths(job, inputDir);
	    Path outpath = new Path(tmpOutput);
	    FileOutputFormat.setOutputPath(job, outpath);	    
	    MapReduceTool.deleteFileIfExistOnHDFS(outpath, job);
	    
	    //set number of reducers (1 if local mode)
	    if( InfrastructureAnalyzer.isLocalMode(job) )
	    	job.setNumReduceTasks(1);
	    else
	    	MRJobConfiguration.setNumReducers(job, numReducers, numReducers);
	    
	    //setup input/output format
	    job.setInputFormat(SamplingSortMRInputFormat.class);
	    SamplingSortMRInputFormat.setTargetKeyValueClasses(job, (Class<? extends WritableComparable>) outputInfo.outputKeyClass, outputInfo.outputValueClass);
	    
	    //setup instructions and meta information
	    if(combineInst!=null&&!combineInst.trim().isEmpty())
	    	job.set(COMBINE_INSTRUCTION, combineInst);
	    job.set(SORT_INSTRUCTION, sortInst);
	    job.setBoolean(VALUE_IS_WEIGHT, valueIsWeight);
	    boolean desc = getSortInstructionDescending(sortInst);
	    job.setBoolean(SORT_DECREASING, desc);
	    MRJobConfiguration.setBlockSize(job, (byte)0, brlen, bclen);
	    MRJobConfiguration.setInputInfo(job, (byte)0, inputInfo, brlen, bclen, ConvertTarget.CELL);
	    int partitionWith0=SamplingSortMRInputFormat.writePartitionFile(job, partitionFile);
	    
	    //setup mapper/reducer/partitioner/output classes
	    if( getSortInstructionType(sortInst)==SortKeys.OperationTypes.Indexes ){
		    MRJobConfiguration.setInputInfo(job, (byte)0, inputInfo, brlen, bclen, ConvertTarget.CELL);
		    job.setOutputFormat(OutputInfo.BinaryBlockOutputInfo.outputFormatClass);
	    	job.setMapperClass(IndexSortMapper.class);
		    job.setReducerClass(IndexSortReducer.class);
		    job.setMapOutputKeyClass( !desc ? IndexSortComparable.class : IndexSortComparableDesc.class);
		    job.setMapOutputValueClass(LongWritable.class);		    
		    job.setOutputKeyClass(MatrixIndexes.class); 
		    job.setOutputValueClass(MatrixBlock.class);   
	    }
	    else { //default case: SORT w/wo weights
	    	MRJobConfiguration.setInputInfo(job, (byte)0, inputInfo, brlen, bclen, ConvertTarget.CELL);
	    	job.setOutputFormat(CompactOutputFormat.class);
		    job.setMapperClass(ValueSortMapper.class);
		    job.setReducerClass(ValueSortReducer.class);	
		    job.setOutputKeyClass(outputInfo.outputKeyClass); //double
		    job.setOutputValueClass(outputInfo.outputValueClass); //int
	    }
	    job.setPartitionerClass(TotalOrderPartitioner.class);
	    
	    
	    //setup distributed cache
	    DistributedCache.addCacheFile(partitionUri, job);
	    DistributedCache.createSymlink(job);
	    
	    //setup replication factor
	    job.setInt("dfs.replication", replication);
	    
		MatrixCharacteristics[] s = new MatrixCharacteristics[1];
		s[0] = new MatrixCharacteristics(rlen, clen, brlen, bclen);
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompleteMRJobInstruction(s);
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		
		//run mr job
	    RunningJob runjob=JobClient.runJob(job);
		Group group=runjob.getCounters().getGroup(NUM_VALUES_PREFIX);
		numReducers=job.getNumReduceTasks();
		
		//process final meta data
		long[] counts=new long[numReducers];
		long total=0;
		for(int i=0; i<numReducers; i++) {
			counts[i]=group.getCounter(Integer.toString(i));
			total+=counts[i];
		}
		
		//add missing 0s back to the results
		long missing0s=0;
		if(total<rlen*clen)
		{
			if(partitionWith0<0) throw new RuntimeException("no partition contains 0, which is wrong!");
			missing0s=rlen*clen-total;
			counts[partitionWith0]+=missing0s;
		}else
			partitionWith0=-1;
		
		if( sortIndexes ) {
			//run builtin job for shifting partially sorted blocks according to global offsets
			//we do this in this custom form since it would not fit into the current structure
			//of systemml to output two intermediates (partially sorted data, offsets) out of a 
			//single SortKeys lop
			boolean success = runjob.isSuccessful();
			if( success ) {
				success = runStitchupJob(tmpOutput, rlen, clen, brlen, bclen, counts, numReducers, replication, output);
			}
			MapReduceTool.deleteFileIfExistOnHDFS( tmpOutput );	
			MapReduceTool.deleteFileIfExistOnHDFS( pfname );
			return new JobReturn(s[0], OutputInfo.BinaryBlockOutputInfo, success);
		}
		else
		{
			MapReduceTool.deleteFileIfExistOnHDFS( pfname );
		    return new JobReturn(s[0], counts, partitionWith0, missing0s, runjob.isSuccessful());
		}
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 */
	private static SortKeys.OperationTypes getSortInstructionType(String str)
	{
		String[] parts = str.split(Lop.OPERAND_DELIMITOR);
		return SortKeys.OperationTypes.valueOf(parts[parts.length-2]);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 */
	private static boolean getSortInstructionDescending(String str)
	{
		String[] parts = str.split(Lop.OPERAND_DELIMITOR);
		return Boolean.parseBoolean(parts[5]);
	}
	
	/**
	 * 
	 * @param input
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param counts
	 * @param numReducers
	 * @param replication
	 * @param output
	 * @throws Exception
	 */
	private static boolean runStitchupJob(String input, long rlen, long clen, int brlen, int bclen, long[] counts,
			int numReducers, int replication, String output) 
	  throws Exception 
	  {
	    JobConf job = new JobConf(SortMR.class);
	    job.setJobName("SortIndexesMR");
	   
	    //setup input/output paths
	    Path inpath = new Path(input);
	    Path outpath = new Path(output);
	    FileInputFormat.setInputPaths(job, inpath);
	    FileOutputFormat.setOutputPath(job, outpath);	    
	    MapReduceTool.deleteFileIfExistOnHDFS(outpath, job);
	    
	    //set number of reducers (1 if local mode)
	    if( InfrastructureAnalyzer.isLocalMode(job) )
	    	job.setNumReduceTasks(1);
	    else
	    	MRJobConfiguration.setNumReducers(job, numReducers, numReducers);
	    	    
	    //setup input/output format
	    InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
	    OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
	    job.setInputFormat(iinfo.inputFormatClass);
	    job.setOutputFormat(oinfo.outputFormatClass);
	    CompactInputFormat.setKeyValueClasses(job, MatrixIndexes.class, MatrixBlock.class);
	    
	    //setup mapper/reducer/output classes
	    MRJobConfiguration.setInputInfo(job, (byte)0, InputInfo.BinaryBlockInputInfo, brlen, bclen, ConvertTarget.BLOCK);
	    job.setMapperClass(IndexSortStitchupMapper.class);
		job.setReducerClass(IndexSortStitchupReducer.class);	
		job.setOutputKeyClass(oinfo.outputKeyClass);
		job.setOutputValueClass(oinfo.outputValueClass); 
	    MRJobConfiguration.setBlockSize(job, (byte)0, brlen, bclen);
	    MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{rlen}, new long[]{clen});
	    
	    //compute shifted prefix sum of offsets and put into configuration
	    long[] cumsumCounts = new long[counts.length];
	    long sum = 0;
	    for( int i=0; i<counts.length; i++ ) {
	    	cumsumCounts[i] = sum;
	    	sum += counts[i];
	    }
	    job.set(SORT_INDEXES_OFFSETS, Arrays.toString(cumsumCounts));
	    
	    //setup replication factor
	    job.setInt("dfs.replication", replication);
	    
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job);
		
		//run mr job
	    RunningJob runJob = JobClient.runJob(job);
	    
	    return runJob.isSuccessful();
	}
}
