package com.ibm.bi.dml.runtime.matrix;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.Counters.Group;

import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CombineUnaryInstruction;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.matrix.sort.CompactOutputFormat;
import com.ibm.bi.dml.runtime.matrix.sort.PickFromCompactInputFormat;
import com.ibm.bi.dml.runtime.matrix.sort.SamplingSortMRInputFormat;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class SortMR {
  public static final String NUM_VALUES_PREFIX="num.values.in";
  public static final String INSTRUCTION = "instruction.before.sort";
  static final String VALUE_IS_WEIGHT="value.is.weight";
  private static final Log LOG = LogFactory.getLog(SortMR.class.getName());
  /**
   * A partitioner that splits text keys into roughly equal partitions
   * in a global sorted order.
   */
  @SuppressWarnings("unchecked")
static class TotalOrderPartitioner<K extends WritableComparable, V extends Writable> 
  implements Partitioner<K, V>{
   
	private Vector<WritableComparable> splitPoints;
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
    private Vector<WritableComparable> readPartitions(FileSystem fs, Path p, 
                                         JobConf job) throws IOException {
    	SequenceFile.Reader reader = new SequenceFile.Reader(fs, p, job);
    	Vector<WritableComparable> parts = new Vector<WritableComparable>();
    	WritableComparable key;
		try {
			key = keyClass.newInstance();
			NullWritable value = NullWritable.get();
			while (reader.next(key, value)) {
				parts.add(key);
				key=keyClass.newInstance();
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		
		reader.close();
	//	System.out.println("read Paritions: "+parts);
	//	System.out.println("num reducers: "+job.getNumReduceTasks());
		return parts;
    }

    public void configure(JobConf job) {
      try {
    	  keyClass=(Class<? extends WritableComparable>) job.getOutputKeyClass();
          valueClass=(Class<? extends Writable>) job.getOutputValueClass();
        FileSystem fs = FileSystem.get(job);
        Path partFile = new Path(MRJobConfiguration.getSortPartitionFilename(job)); 
        splitPoints = readPartitions(fs, partFile, job);
     //   System.out.println("num reducers: "+job.getNumReduceTasks());
       } catch (IOException ie) {
        throw new IllegalArgumentException("can't read paritions file", ie);
      }
    }

    public TotalOrderPartitioner() {
    }

    public int getPartition(K key, V value, int numPartitions) {
    //	System.out.println("num paritions in getPartition: "+numPartitions);
  //  	System.out.println(key+" is in parititon: "+findPartition(key));
      return findPartition(key)%numPartitions;
    }

	private int findPartition(K key) {
		int i=0;
		for( ; i<splitPoints.size(); i++)
		{
			if(key.compareTo(splitPoints.get(i))<=0)
				return i;
		}
		return i;
	}
    
  }
 
  static class InnerMapper<KIN extends WritableComparable, VIN extends Writable, KOUT extends WritableComparable, VOUT extends Writable> 
  extends MapReduceBase implements Mapper<KIN, VIN, KOUT, VOUT>
  {
	  private int brlen;
	private int bclen;
	//private String instruction;
	private CombineUnaryInstruction combineInstruction=null;
	private Converter<KIN, VIN, KOUT, VOUT> inputConverter;
	private IntWritable one=new IntWritable(1);
	private DoubleWritable combinedKey=new DoubleWritable();
	
	public void map(KIN key, VIN value, OutputCollector<KOUT, VOUT> out,
			Reporter reporter) throws IOException {
		inputConverter.convert(key, value);
		while(inputConverter.hasNext())
		{
			Pair pair=inputConverter.next();
			if(combineInstruction==null)
			{
				//System.out.println("output: "+pair.getKey()+": "+pair.getValue());
				out.collect((KOUT) pair.getKey(), (VOUT)pair.getValue());
			}else
			{
				processCombineUnaryInstruction(pair, out);
			}
		}
	} 
	
	private void processCombineUnaryInstruction(Pair pair, OutputCollector<KOUT, VOUT> out) 
	throws IOException
	{
		combinedKey.set(((MatrixCell)pair.getValue()).getValue());
		out.collect((KOUT)combinedKey, (VOUT)one);
		//System.out.println("output: "+combinedKey+": "+one);
	}
	
	public void configure(JobConf job)
	{
		//System.out.println("enter map.configuration");
		brlen = MRJobConfiguration.getNumRowsPerBlock(job, (byte) 0);
		bclen = MRJobConfiguration.getNumColumnsPerBlock(job, (byte) 0);
		String str=job.get(SortMR.INSTRUCTION);
		if(str!=null && !str.isEmpty())
			try {
				combineInstruction=(CombineUnaryInstruction) CombineUnaryInstruction.parseInstruction(str);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		inputConverter = MRJobConfiguration.getInputConverter(job, (byte) 0);
		inputConverter.setBlockSize(brlen, bclen);
	}
  }
  
  static class InnerReducer<K extends WritableComparable, V extends Writable> extends MapReduceBase implements Reducer<K, V, K, V>
  {
	  private String taskID=null;
	  private boolean valueIsWeight=false;
	  private long count=0;
	
	public void configure(JobConf job)
	{
		taskID=MapReduceTool.getUniqueKeyPerTask(job, false);
		valueIsWeight=job.getBoolean(VALUE_IS_WEIGHT, false);
	}

	@Override
	public void reduce(K key, Iterator<V> values, OutputCollector<K, V> out,
			Reporter report) throws IOException {
		int sum=0;
		while(values.hasNext())
		{
			V value=values.next();
			out.collect(key, value);
		//	LOG.info("** sort result: "+key+": "+value);
			//System.out.println("sort "+count+": "+key+": "+value);
			if(valueIsWeight)
				sum+=((IntWritable)value).get();
			else
				sum++;
		}
		count+=sum;
		report.incrCounter(NUM_VALUES_PREFIX, taskID, sum);
	}
  }
  
	public static JobReturn runJob(MRJobInstruction inst, String input, InputInfo inputInfo, long rlen, long clen, 
			int brlen, int bclen, String instructionBeforesort, int numReducers, 
			int replication, String output, OutputInfo outputInfo, boolean valueIsWeight) 
	  throws Exception 
	  {
		//  if(!inputInfo.inputKeyClass.equals(outputInfo.outputKeyClass) 
		//		  || !inputInfo.inputValueClass.equals(outputInfo.outputValueClass))
		//	  throw new Exception("input key value pair ("+inputInfo.inputKeyClass+", "
		//			  +inputInfo.inputValueClass+") does not match output key value pair ("+outputInfo.outputKeyClass+", "
		//			  +outputInfo.outputValueClass+")!");
		
	    JobConf job = new JobConf(SortMR.class);
	    Path inputDir = new Path(input);
	    inputDir = inputDir.makeQualified(inputDir.getFileSystem(job));
	    
	    //setup partition file
	    String pfname = MRJobConfiguration.setUpSortPartitionFilename(job);
	    Path partitionFile = new Path( pfname ); 
	    URI partitionUri = new URI( partitionFile.toString() + "#" + pfname ); 
	   
	    SamplingSortMRInputFormat.setInputPaths(job, inputDir);
	    Path outpath=new Path(output);
	    FileOutputFormat.setOutputPath(job, outpath);
	    
	    MapReduceTool.deleteFileIfExistOnHDFS(outpath, job);
	    //detect whether this is running in local mode or not, since if local mode, the number of reducers are hard set to 1 no matter what
	//    System.out.println(job.get("fs.default.name"));
	//    System.out.println(job.get("mapred.job.tracker"));
	    if(job.get("mapred.job.tracker").indexOf("local")>=0)
	    	job.setNumReduceTasks(1);
	    else
	    	MRJobConfiguration.setNumReducers(job, numReducers, numReducers);
	    job.setJobName("SortMR");
	    job.setOutputKeyClass(outputInfo.outputKeyClass);
	    job.setOutputValueClass(outputInfo.outputValueClass);
	  //  job.setMapOutputValueClass(WeightedCell.class);
	    job.setInputFormat(SamplingSortMRInputFormat.class);
	    SamplingSortMRInputFormat.setTargetKeyValueClasses(job, (Class<? extends WritableComparable>) outputInfo.outputKeyClass, outputInfo.outputValueClass);
	    job.setOutputFormat(CompactOutputFormat.class);
	    job.setPartitionerClass(TotalOrderPartitioner.class);
	    job.setMapperClass(InnerMapper.class);
	    job.setReducerClass(InnerReducer.class);
	    if(instructionBeforesort!=null)
	    	job.set(INSTRUCTION, instructionBeforesort);
	    MRJobConfiguration.setBlockSize(job, (byte)0, brlen, bclen);
	    MRJobConfiguration.setInputInfo(job, (byte)0, inputInfo, false, brlen, bclen);
	    int partitionWith0=SamplingSortMRInputFormat.writePartitionFile(job, partitionFile);
	    DistributedCache.addCacheFile(partitionUri, job);
	    DistributedCache.createSymlink(job);
	    job.setInt("dfs.replication", replication);
	  //  System.out.println("num reducers: "+job.getNumReduceTasks());
	    job.setBoolean(VALUE_IS_WEIGHT, valueIsWeight);
	    
		MatrixCharacteristics s[] = new MatrixCharacteristics[1];
		s[0] = new MatrixCharacteristics(rlen, clen, brlen, bclen);
		
	    // By default, the job executes in "cluster" mode.
		// Determine if we can optimize and run it in "local" mode.
	    ExecMode mode = RunMRJobs.getExecMode(JobType.SORT, s); 
		if ( mode == ExecMode.LOCAL ) {
			job.set("mapred.job.tracker", "local");
			MRJobConfiguration.setStagingDir( job );
		}
		
		// Print the complete instruction
		if (LOG.isTraceEnabled())
			inst.printCompelteMRJobInstruction(s);
		
		//set unique working dir
		MRJobConfiguration.setUniqueWorkingDir(job, mode);
		
		
	    RunningJob runjob=JobClient.runJob(job);
		Group group=runjob.getCounters().getGroup(NUM_VALUES_PREFIX);
		numReducers=job.getNumReduceTasks();
	//	System.out.println("num reducers: "+job.getNumReduceTasks());
		
		long[] counts=new long[numReducers];
		long total=0;
		for(int i=0; i<numReducers; i++)
		{
			counts[i]=group.getCounter(Integer.toString(i));
			total+=counts[i];
		//	System.out.println("partition "+i+": "+counts[i]);
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
		
	//	runSelect(output, "some", counts, new double[]{0.4, 0.2, 0.41, 0.7});
	//	runSelect(output, "some", counts, 0.25, 0.75);//new double[]{0.4, 0.2, 0.45, 0.7});
	  //  LOG.info("done");
	    MapReduceTool.deleteFileIfExistOnHDFS( pfname );
	    return new JobReturn(s[0], counts, partitionWith0, missing0s, runjob.isSuccessful());
	  }

  /*
  public static void runSelect(String input, String output, long[] counts, double[] probs) throws IOException
  {
	  	JobConf job = new JobConf(SortMR.class);
	    Path inputDir = new Path(input);
	    SamplingSortMRInputFormat.setInputPaths(job, inputDir);
	    Path outpath=new Path(output);
	    FileOutputFormat.setOutputPath(job, outpath);
	    
	    MapReduceTool.deleteFileIfExistOnHDFS(outpath, job);
	    job.setNumReduceTasks(0);
	    job.setJobName("select");
	    job.setOutputKeyClass(MatrixIndexes.class);
	    job.setOutputValueClass(MatrixCell.class);
	    PickFromCompactInputFormat.setKeyValueClasses(job, DoubleWritable.class, IntWritable.class);
	    job.setInputFormat(PickFromCompactInputFormat.class);
	    PickFromCompactInputFormat.setPickRecordsInEachPartFile(job, counts, probs);
	    job.setMapperClass(IdentityMapper.class);
	    RunningJob runjob=JobClient.runJob(job);
  }
  
  public static void runSelect(String input, String output, long[] counts, double lbound, double ubound) throws IOException
  {
	  	JobConf job = new JobConf(SortMR.class);
	    Path inputDir = new Path(input);
	    SamplingSortMRInputFormat.setInputPaths(job, inputDir);
	    Path outpath=new Path(output);
	    FileOutputFormat.setOutputPath(job, outpath);
	    
	    MapReduceTool.deleteFileIfExistOnHDFS(outpath, job);
	    job.setNumReduceTasks(0);
	    job.setJobName("select");
	    job.setOutputKeyClass(MatrixIndexes.class);
	    job.setOutputValueClass(MatrixCell.class);
	    PickFromCompactInputFormat.setKeyValueClasses(job, DoubleWritable.class, IntWritable.class);
	    job.setInputFormat(PickFromCompactInputFormat.class);
	    PickFromCompactInputFormat.setPickRecordsInEachPartFile(job, counts, lbound, ubound);
	    job.setMapperClass(IdentityMapper.class);
	    RunningJob runjob=JobClient.runJob(job);
  }
  */
  
  /**
   * @param args
   */
  
  public static void testWithoutWeights(String[] args)throws Exception 
  {
	  if(args.length<4)
	  {
		  System.err.println("<input> <output> <numReducers> <selectVector>");
		  System.exit(-1);
	  }
    
    InputInfo inputinfo=new InputInfo(SequenceFileInputFormat.class, DoubleWritable.class, IntWritable.class);
    OutputInfo outputInfo=new OutputInfo(CompactOutputFormat.class, DoubleWritable.class, IntWritable.class);
    
    MRJobInstruction sort_dummy = new MRJobInstruction(JobType.SORT);
    JobReturn ret=runJob(sort_dummy, args[0], InputInfo.BinaryBlockInputInfo, 100, 1, 10, 1, "combineunary:::0:DOUBLE:::1:DOUBLE", 
    		Integer.parseInt(args[2]), 1, args[1], outputInfo, false);
   // System.out.println(MapReduceTool.pickValue(args[1], (NumItemsByEachReducerMetaData) ret.otherMetadata[0], 0.5));
    inputinfo.inputFormatClass=PickFromCompactInputFormat.class;
    inputinfo.inputKeyClass=DoubleWritable.class;
    inputinfo.inputValueClass=IntWritable.class;
    inputinfo.metadata=ret.getMetaData(0);
    
    String rrins="valuepick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
    
    MatrixCharacteristics matchar = ((MatrixDimensionsMetaData)ret.getMetaData(0)).getMatrixCharacteristics();
    MRJobInstruction dummy = new MRJobInstruction(JobType.GMR);
    GMR.runJob(dummy, new String[]{args[1], args[3]}, new InputInfo[]{inputinfo, InputInfo.TextCellInputInfo}, 
    		new long[]{matchar.numRows, 4}, new long[]{matchar.numColumns, 4}, new int[]{matchar.numRowsPerBlock, 1}, 
    		new int[]{matchar.numColumnsPerBlock, 1}, null, null, null, rrins, "", "", "", 0, 1, 
    		new byte[]{2}, "scratch_space", new String[]{"final"}, new OutputInfo[]{OutputInfo.TextCellOutputInfo});
    
    rrins="rangepick:::0:DOUBLE:::0.25:DOUBLE:::2:DOUBLE";
    GMR.runJob(dummy, new String[]{args[1]}, new InputInfo[]{inputinfo}, 
    		new long[]{matchar.numRows}, new long[]{matchar.numColumns}, new int[]{matchar.numRowsPerBlock}, 
    		new int[]{matchar.numColumnsPerBlock}, null, null, null, rrins, "", "", "", 0, 1, 
    		new byte[]{2}, "scratch_space", new String[]{"final"}, new OutputInfo[]{OutputInfo.TextCellOutputInfo});
  }
  
  public static void testWithWeights(String[] args)throws Exception 
  {
	  if(args.length<5)
	  {
		  System.err.println("<input> <inputweight> <output> <numReducers> <selectVector>");
		  System.exit(-1);
	  }
	  
    OutputInfo outputInfo=new OutputInfo(SequenceFileOutputFormat.class, DoubleWritable.class, IntWritable.class);
    
    MRJobInstruction dummy = new MRJobInstruction(JobType.COMBINE);
    CombineMR.runJob(dummy, new String[]{args[0], args[1]}, new InputInfo[]{InputInfo.BinaryBlockInputInfo, InputInfo.BinaryBlockInputInfo}, 
    		new long[]{100, 100}, new long[]{1, 1}, new int[]{10, 10}, new int[]{1, 1}, 
    		"combinebinary:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE", Integer.parseInt(args[3]), 1, new byte[]{2}, 
    		new String[]{"temp"}, new OutputInfo[]{outputInfo});
    
    InputInfo inputinfo=new InputInfo(SequenceFileInputFormat.class, DoubleWritable.class, IntWritable.class);
    outputInfo=new OutputInfo(CompactOutputFormat.class, DoubleWritable.class, IntWritable.class);
    MRJobInstruction sort_dummy = new MRJobInstruction(JobType.SORT);
    JobReturn ret=runJob(sort_dummy, "temp", inputinfo, 100, 1, 10, 1, null, Integer.parseInt(args[3]), 1, args[2], outputInfo, true);

   // System.out.println(MapReduceTool.pickValue(args[2], (NumItemsByEachReducerMetaData) ret.otherMetadata[0], 0.5));
    inputinfo.inputFormatClass=PickFromCompactInputFormat.class;
    inputinfo.inputKeyClass=DoubleWritable.class;
    inputinfo.inputValueClass=IntWritable.class;
    inputinfo.metadata=ret.getMetaData(0);
    
    String rrins="valuepick:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE";
    
    MatrixCharacteristics matchar = ((MatrixDimensionsMetaData)ret.getMetaData(0)).getMatrixCharacteristics();
    dummy = new MRJobInstruction(JobType.GMR);

    GMR.runJob(dummy, new String[]{args[2], args[4]}, new InputInfo[]{inputinfo, InputInfo.TextCellInputInfo}, 
    		new long[]{matchar.numRows, 4}, new long[]{matchar.numColumns, 4}, new int[]{matchar.numRowsPerBlock, 1}, 
    		new int[]{matchar.numColumnsPerBlock, 1}, null, null, null, rrins, "", "", "", 0, 1, 
    		new byte[]{2}, "scratch_space", new String[]{"final"}, new OutputInfo[]{OutputInfo.TextCellOutputInfo});
    
    rrins="rangepick:::0:DOUBLE:::0.25:DOUBLE:::2:DOUBLE";
    GMR.runJob(dummy, new String[]{args[2]}, new InputInfo[]{inputinfo}, 
    		new long[]{matchar.numRows}, new long[]{matchar.numColumns}, new int[]{matchar.numRowsPerBlock}, 
    		new int[]{matchar.numColumnsPerBlock}, null, null, null, rrins, "", "", "", 0, 1, 
    		new byte[]{2}, "scratch_space", new String[]{"final"}, new OutputInfo[]{OutputInfo.TextCellOutputInfo});
  }

public static void main(String[] args) throws Exception {
	  testWithoutWeights(args);
  }
}
