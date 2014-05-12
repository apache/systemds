/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.NullOutputFormat;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.PairWritableCell;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.Statistics;

/**
 * MR job class for submitting parfor remote partitioning MR jobs.
 *
 */
public class DataPartitionerRemoteMR extends DataPartitioner
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _pfid = -1;
	//private int  _numMappers = -1;
	private int  _numReducers = -1;
	private int  _replication = -1;
	private int  _max_retry = -1;
	private boolean _jvmReuse = false;

	public DataPartitionerRemoteMR(PDataPartitionFormat dpf, int n) 
	{
		this( dpf, n, -1, 
			  Math.min( ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS),
						InfrastructureAnalyzer.getRemoteParallelReduceTasks() ),
		      1, 3, false );
	}
	
	public DataPartitionerRemoteMR(PDataPartitionFormat dpf, int n, long pfid, int numReducers, int replication, int max_retry, boolean jvmReuse) 
	{
		super(dpf, n);
		
		_pfid = pfid;
		_numReducers = numReducers;
		_replication = replication;
		_max_retry = max_retry;
		_jvmReuse = jvmReuse;
	}


	@Override
	protected void partitionMatrix(String fname, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen)
			throws DMLRuntimeException 
	{
		String jobname = "ParFor-DPMR";
		long t0 = System.nanoTime();
		
		JobConf job;
		job = new JobConf( DataPartitionerRemoteMR.class );
		if( _pfid >= 0 ) //use in parfor
			job.setJobName(jobname+_pfid);
		else //use for partition instruction
			job.setJobName("Partition-MR");
			
		//maintain dml script counters
		Statistics.incrementNoOfCompiledMRJobs();
		
		try
		{
			Path path = new Path(fname);
			
			/////
			//configure the MR job
			MRJobConfiguration.setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, _format, _n, fnameNew);
			
			//set mappers, reducers, combiners
			job.setMapperClass(DataPartitionerRemoteMapper.class); 
			job.setReducerClass(DataPartitionerRemoteReducer.class);
			
			if( oi == OutputInfo.TextCellOutputInfo )
			{
				//binary cell intermediates for reduced IO 
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(PairWritableCell.class);	
			}
			else if( oi == OutputInfo.BinaryCellOutputInfo )
			{
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(PairWritableCell.class);
			}
			else if ( oi == OutputInfo.BinaryBlockOutputInfo )
			{
				job.setMapOutputKeyClass(LongWritable.class);
				job.setMapOutputValueClass(PairWritableBlock.class);
				
				//check Alignment
				if(   (_format == PDataPartitionFormat.ROW_BLOCK_WISE_N && rlen>_n && _n % brlen !=0)
        			|| (_format == PDataPartitionFormat.COLUMN_BLOCK_WISE_N && clen>_n && _n % bclen !=0) )
				{
					throw new DMLRuntimeException("Data partitioning format "+_format+" requires aligned blocks.");
				}
			}
			
			//set input format 
			job.setInputFormat(ii.inputFormatClass);
			
			//set the input path and output path 
		    FileInputFormat.setInputPaths(job, path);
			
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(fnameNew);
		    //FileOutputFormat.setOutputPath(job, pathNew);
		    job.setOutputFormat(NullOutputFormat.class);

			//////
			//set optimization parameters

			//set the number of mappers and reducers 
		    //job.setNumMapTasks( _numMappers ); //use default num mappers
		    long reducerGroups = -1;
		    switch( _format )
		    {
			    case ROW_WISE: reducerGroups = rlen; break;
			    case COLUMN_WISE: reducerGroups = clen; break;
			    case ROW_BLOCK_WISE: reducerGroups = (rlen/brlen)+((rlen%brlen==0)?0:1); break;
			    case COLUMN_BLOCK_WISE: reducerGroups = (clen/bclen)+((clen%bclen==0)?0:1); break;
			    case ROW_BLOCK_WISE_N: reducerGroups = (rlen/_n)+((rlen%_n==0)?0:1); break;
			    case COLUMN_BLOCK_WISE_N: reducerGroups = (clen/_n)+((clen%_n==0)?0:1); break;
		    }
		    job.setNumReduceTasks( (int)Math.min( _numReducers, reducerGroups) ); 	

			//use FLEX scheduler configuration properties
			/*if( ParForProgramBlock.USE_FLEX_SCHEDULER_CONF )
			{
				job.setInt("flex.map.min", 0);
				job.setInt("flex.map.max", _numMappers);
				job.setInt("flex.reduce.min", 0);
				job.setInt("flex.reduce.max", _numMappers);
			}*/
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt("mapred.task.timeout", 0);			
			job.setMapSpeculativeExecution(false);
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( _jvmReuse )
				job.setNumTasksToExecutePerJvm(-1); //unlimited
			
			//enables compression - not conclusive for different codecs (empirically good compression ratio, but significantly slower)
			//job.set("mapred.compress.map.output", "true");
			//job.set("mapred.map.output.compression.codec", "org.apache.hadoop.io.compress.GzipCodec");
			
			//set the replication factor for the results
			job.setInt("dfs.replication", _replication);
			
			//set the max number of retries per map task
			//  disabled job-level configuration to respect cluster configuration
			//  note: this refers to hadoop2, hence it never had effect on mr1
			//job.setInt("mapreduce.map.maxattempts", _max_retry);
			
			//set unique working dir
			ExecMode mode = ExecMode.CLUSTER;
			MRJobConfiguration.setUniqueWorkingDir(job, mode);
			
			/////
			// execute the MR job	
			JobClient.runJob(job);
		
			//maintain dml script counters
			Statistics.incrementNoOfExecutedMRJobs();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		if( DMLScript.STATISTICS ){
			long t1 = System.nanoTime();
			Statistics.maintainCPHeavyHitters("MR-Job_"+jobname, t1-t0);
		}
	}
	
}
