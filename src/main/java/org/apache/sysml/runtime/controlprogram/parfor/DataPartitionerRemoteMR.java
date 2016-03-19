/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.NullOutputFormat;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableCell;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.yarn.DMLAppMasterUtils;

/**
 * MR job class for submitting parfor remote partitioning MR jobs.
 *
 */
public class DataPartitionerRemoteMR extends DataPartitioner
{	
	
	private long _pfid = -1;
	private int  _numReducers = -1;
	private int  _replication = -1;
	//private int  _max_retry = -1;
	private boolean _jvmReuse = false;
	private boolean _keepIndexes = false;
	
	
	public DataPartitionerRemoteMR(PDataPartitionFormat dpf, int n, long pfid, int numReducers, int replication, int max_retry, boolean jvmReuse, boolean keepIndexes) 
	{
		super(dpf, n);
		
		_pfid = pfid;
		_numReducers = numReducers;
		_replication = replication;
		//_max_retry = max_retry;
		_jvmReuse = jvmReuse;
		_keepIndexes = keepIndexes;
	}


	@Override
	protected void partitionMatrix(MatrixObject in, String fnameNew, InputInfo ii, OutputInfo oi, long rlen, long clen, int brlen, int bclen)
			throws DMLRuntimeException 
	{
		String jobname = "ParFor-DPMR";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
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
			//force writing to disk (typically not required since partitioning only applied if dataset exceeds CP size)
			in.exportData(); //written to disk iff dirty
			
			Path path = new Path(in.getFileName());
			
			/////
			//configure the MR job
			MRJobConfiguration.setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, _format, _n, fnameNew, _keepIndexes);
			
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
			    default:
					//do nothing
		    }
		    job.setNumReduceTasks( (int)Math.min( _numReducers, reducerGroups) ); 	

			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt(MRConfigurationNames.MR_TASK_TIMEOUT, 0);
			job.setMapSpeculativeExecution(false);
			
			//set up preferred custom serialization framework for binary block format
			if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
				MRJobConfiguration.addBinaryBlockSerializationFramework( job );
			
			//enables the reuse of JVMs (multiple tasks per MR task)
			if( _jvmReuse )
				job.setNumTasksToExecutePerJvm(-1); //unlimited
			
			//enables compression - not conclusive for different codecs (empirically good compression ratio, but significantly slower)
			//job.set(MRConfigurationNames.MR_MAP_OUTPUT_COMPRESS, "true");
			//job.set(MRConfigurationNames.MR_MAP_OUTPUT_COMPRESS_CODEC, "org.apache.hadoop.io.compress.GzipCodec");
			
			//set the replication factor for the results
			job.setInt(MRConfigurationNames.DFS_REPLICATION, _replication);
			
			//set up map/reduce memory configurations (if in AM context)
			DMLConfig config = ConfigurationManager.getDMLConfig();
			DMLAppMasterUtils.setupMRJobRemoteMaxMemory(job, config);
			
			//set up custom map/reduce configurations 
			MRJobConfiguration.setupCustomMRConfigurations(job, config);
			
			//set the max number of retries per map task
			//  disabled job-level configuration to respect cluster configuration
			//  note: this refers to hadoop2, hence it never had effect on mr1
			//job.setInt(MRConfigurationNames.MR_MAP_MAXATTEMPTS, _max_retry);
			
			//set unique working dir
			MRJobConfiguration.setUniqueWorkingDir(job);
			
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
		
		if( DMLScript.STATISTICS && _pfid >= 0 ){ 
			long t1 = System.nanoTime(); //only for parfor 
			Statistics.maintainCPHeavyHitters("MR-Job_"+jobname, t1-t0);
		}
	}
	
}
