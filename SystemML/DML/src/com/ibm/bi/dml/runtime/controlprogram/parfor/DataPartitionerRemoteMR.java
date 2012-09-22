package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.LinkedList;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 * 
 * TODO create or handle not created directories.
 *
 */
public class DataPartitionerRemoteMR extends DataPartitioner
{
	private long _pfid = -1;
	private int  _numMappers = -1;
	private int  _replication = -1;
	private int  _max_retry = -1;
	private boolean _jvmReuse = false;

	public DataPartitionerRemoteMR(PDataPartitionFormat dpf, long pfid, int numMappers, int replication, int max_retry, boolean jvmReuse) 
	{
		super(dpf);
		
		_pfid = pfid;
		_numMappers = numMappers;
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
		job.setJobName("ParFor_Partition-MR_"+_pfid);
		
		try
		{
			Path path = new Path(fname);
			Path pathNew = new Path(fnameNew);
			
			/////
			//configure the MR job
			MRJobConfiguration.setPartitioningInfoInMapper(job, rlen, clen, brlen, bclen, ii, _format);
			
			//set mappers, reducers, combiners
			job.setMapperClass(DataPartitionerRemoteMapper.class); //map-only

			//set input format (one split per row, NLineInputFormat default N=1)
			job.setInputFormat(ii.inputFormatClass);
			
			//set the input path and output path 
		    FileInputFormat.setInputPaths(job, path);
			
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
		    FileOutputFormat.setOutputPath(job, pathNew);
		    
		    //create named outputs, set output format, and set the output key, value schema
		    LinkedList<Long> outputFiles = new LinkedList<Long>(); //TODO potential memory bottleneck
		    switch( _format )
		    {
			    case ROW_WISE:
			    	for( long i=1; i<=rlen; i++ )
			    		outputFiles.addLast( i );
			    	break;
			    case ROW_BLOCK_WISE:
			    	for( long i=1; i<=rlen/brlen+1; i++ )
			    		outputFiles.addLast( i );
			    	break;
			    case COLUMN_WISE:
			    	for( long j=1; j<=clen; j++ )
			    		outputFiles.addLast( j );
			    	break;
			    case COLUMN_BLOCK_WISE:
			    	for( long j=1; j<=clen/bclen+1; j++ )
			    		outputFiles.addLast( j );
			    	break;
		    }
		    
		    for( Long out : outputFiles ) 
		    {
		    	MultipleOutputs.addNamedOutput(job, String.valueOf(out), oi.outputFormatClass, oi.outputKeyClass, oi.outputValueClass);
		    }

			//////
			//set optimization parameters

			//set the number of mappers and reducers 
			job.setNumMapTasks(_numMappers); //numMappers
			job.setNumReduceTasks( 0 );			

			//use FLEX scheudler configuration properties
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

			
			//postprocessing (required because MultipleOutputs creates concatenated filenames)
			//TODO change to direct access in order to prevent memory bottleneck
			FileSystem fs = FileSystem.get(job);
			FileStatus[] status = fs.listStatus(pathNew);
			for( FileStatus f : status )
			{
				String lname = f.getPath().getName(); 
				if( lname.startsWith("_") || lname.startsWith(".") ) //reject system internal files
					continue;
				String lnameNew = lname.split("-")[0];
				fs.mkdirs( new Path(fnameNew+"/"+lnameNew) ); //create dir
				fs.rename(new Path(fnameNew+"/"+lname), new Path(fnameNew+"/"+lnameNew+"/"+lname));
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}		
	}
}
