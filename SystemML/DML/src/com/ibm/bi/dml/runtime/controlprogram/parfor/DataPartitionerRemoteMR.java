package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 *
 */
public class DataPartitionerRemoteMR extends DataPartitioner
{	
	private long _pfid = -1;
	private int  _numMappers = -1;
	private int  _numReducers = -1;
	private int  _replication = -1;
	private int  _max_retry = -1;
	private boolean _jvmReuse = false;

	public DataPartitionerRemoteMR(PDataPartitionFormat dpf, long pfid, int numMappers, int numReducers, int replication, int max_retry, boolean jvmReuse) 
	{
		super(dpf);
		
		_pfid = pfid;
		_numMappers = numMappers;
		_numReducers = numReducers;
		_replication = replication;
		_max_retry = max_retry;
		_jvmReuse = jvmReuse;
	}


	@Override
	protected void partitionMatrix(String fname, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen)
			throws DMLRuntimeException 
	{
		JobConf job;
		job = new JobConf( DataPartitionerRemoteMR.class );
		job.setJobName("ParFor_Partition-MR"+_pfid);
		
		try
		{
			Path path = new Path(fname);
			Path pathNew = new Path(fnameNew);
			
			/////
			//configure the MR job
			MRJobConfiguration.setPartitioningInfoInMapper(job, rlen, clen, brlen, bclen, ii, _format, fnameNew);
			
			//set mappers, reducers, combiners
			job.setMapperClass(DataPartitionerRemoteMapper.class); 
			job.setReducerClass(DataPartitionerRemoteReducer.class);
			
			if( ii == InputInfo.TextCellInputInfo )
			{
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(Text.class);	
			}
			else if( ii == InputInfo.BinaryCellInputInfo )
			{
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(PairWritableCell.class);
			}
			else if ( ii == InputInfo.BinaryBlockInputInfo )
			{
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(PairWritableBlock.class);
			}
			
			//set input format (one split per row, NLineInputFormat default N=1)
			job.setInputFormat(ii.inputFormatClass);
			
			//set the input path and output path 
		    FileInputFormat.setInputPaths(job, path);
			
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
		    FileOutputFormat.setOutputPath(job, pathNew);

			//////
			//set optimization parameters

			//set the number of mappers and reducers 
		    //job.setNumMapTasks( _numMappers ); //use default num mappers
			job.setNumReduceTasks( _numReducers ); 			

			//use FLEX scheduler configuration properties
			//System.out.println("numMappers="+numMappers);
			if( ParForProgramBlock.USE_FLEX_SCHEDULER_CONF )
			{
				job.setInt("flex.map.min", 0);
				job.setInt("flex.map.max", _numMappers);
				job.setInt("flex.reduce.min", 0);
				job.setInt("flex.reduce.max", _numMappers);
			}
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( _jvmReuse )
				job.setNumTasksToExecutePerJvm(-1); //unlimited
			
			//set the replication factor for the results
			job.setInt("dfs.replication", _replication);
			
			//set the max number of retries per map task
			job.setInt("mapreduce.map.maxattempts", _max_retry);
			
			//set unique working dir
			ExecMode mode = ExecMode.CLUSTER;
			MRJobConfiguration.setUniqueWorkingDir(job, mode);
			
			/////
			// execute the MR job			
			JobClient.runJob(job);
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}		
	}
	
}
