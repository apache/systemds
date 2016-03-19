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

import java.io.IOException;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters.Group;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Stat;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableBlock;
import org.apache.sysml.runtime.controlprogram.parfor.util.PairWritableCell;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.utils.Statistics;
import org.apache.sysml.yarn.DMLAppMasterUtils;

/**
 * MR job class for submitting parfor remote MR jobs, controlling its execution and obtaining results.
 * 
 *
 */
public class RemoteDPParForMR
{
	
	protected static final Log LOG = LogFactory.getLog(RemoteDPParForMR.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param taskFile
	 * @param resultFile
	 * @param enableCPCaching 
	 * @param mode
	 * @param numMappers
	 * @param replication
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RemoteParForJobReturn runJob(long pfid, String itervar, String matrixvar, String program, String resultFile, MatrixObject input, 
			                                   PDataPartitionFormat dpf, OutputInfo oi, boolean tSparseCol, //config params
			                                   boolean enableCPCaching, int numReducers, int replication, int max_retry)  //opt params
		throws DMLRuntimeException
	{
		RemoteParForJobReturn ret = null;
		String jobname = "ParFor-DPEMR";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		JobConf job;
		job = new JobConf( RemoteDPParForMR.class );
		job.setJobName(jobname+pfid);
		
		//maintain dml script counters
		Statistics.incrementNoOfCompiledMRJobs();
	
		try
		{
			/////
			//configure the MR job
		
			//set arbitrary CP program blocks that will perform in the reducers
			MRJobConfiguration.setProgramBlocks(job, program); 

			//enable/disable caching
			MRJobConfiguration.setParforCachingConfig(job, enableCPCaching);
		
			//setup input matrix
			Path path = new Path( input.getFileName() );
			long rlen = input.getNumRows();
			long clen = input.getNumColumns();
			int brlen = (int) input.getNumRowsPerBlock();
			int bclen = (int) input.getNumColumnsPerBlock();
			MRJobConfiguration.setPartitioningInfo(job, rlen, clen, brlen, bclen, InputInfo.BinaryBlockInputInfo, oi, dpf, 1, input.getFileName(), itervar, matrixvar, tSparseCol);
			job.setInputFormat(InputInfo.BinaryBlockInputInfo.inputFormatClass);
			FileInputFormat.setInputPaths(job, path);
			
			//set mapper and reducers classes
			job.setMapperClass(DataPartitionerRemoteMapper.class); 
			job.setReducerClass(RemoteDPParWorkerReducer.class); 
			
		    //set output format
		    job.setOutputFormat(SequenceFileOutputFormat.class);
		    
		    //set output path
		    MapReduceTool.deleteFileIfExistOnHDFS(resultFile);
		    FileOutputFormat.setOutputPath(job, new Path(resultFile));
		    
			//set the output key, value schema
		    
		    //parfor partitioning outputs (intermediates)
		    job.setMapOutputKeyClass(LongWritable.class);
		    if( oi == OutputInfo.BinaryBlockOutputInfo )
		    	job.setMapOutputValueClass(PairWritableBlock.class); 
		    else if( oi == OutputInfo.BinaryCellOutputInfo )
		    	job.setMapOutputValueClass(PairWritableCell.class);
		    else 
		    	throw new DMLRuntimeException("Unsupported intermrediate output info: "+oi);
		    //parfor exec output
		    job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(Text.class);
			
			//////
			//set optimization parameters

			//set the number of mappers and reducers 
			job.setNumReduceTasks( numReducers );
			
			//disable automatic tasks timeouts and speculative task exec
			job.setInt(MRConfigurationNames.MR_TASK_TIMEOUT, 0);
			job.setMapSpeculativeExecution(false);
			
			//set up preferred custom serialization framework for binary block format
			if( MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION )
				MRJobConfiguration.addBinaryBlockSerializationFramework( job );
	
			//set up map/reduce memory configurations (if in AM context)
			DMLConfig config = ConfigurationManager.getDMLConfig();
			DMLAppMasterUtils.setupMRJobRemoteMaxMemory(job, config);
			
			//set up custom map/reduce configurations 
			MRJobConfiguration.setupCustomMRConfigurations(job, config);
			
			//disable JVM reuse
			job.setNumTasksToExecutePerJvm( 1 ); //-1 for unlimited 
			
			//set the replication factor for the results
			job.setInt(MRConfigurationNames.DFS_REPLICATION, replication);
			
			//set the max number of retries per map task
			//note: currently disabled to use cluster config
			//job.setInt(MRConfigurationNames.MR_MAP_MAXATTEMPTS, max_retry);
			
			//set unique working dir
			MRJobConfiguration.setUniqueWorkingDir(job);
			
			/////
			// execute the MR job			
			RunningJob runjob = JobClient.runJob(job);
			
			// Process different counters 
			Statistics.incrementNoOfExecutedMRJobs();
			Group pgroup = runjob.getCounters().getGroup(ParForProgramBlock.PARFOR_COUNTER_GROUP_NAME);
			int numTasks = (int)pgroup.getCounter( Stat.PARFOR_NUMTASKS.toString() );
			int numIters = (int)pgroup.getCounter( Stat.PARFOR_NUMITERS.toString() );
			if( DMLScript.STATISTICS && !InfrastructureAnalyzer.isLocalMode() ) {
				Statistics.incrementJITCompileTime( pgroup.getCounter( Stat.PARFOR_JITCOMPILE.toString() ) );
				Statistics.incrementJVMgcCount( pgroup.getCounter( Stat.PARFOR_JVMGC_COUNT.toString() ) );
				Statistics.incrementJVMgcTime( pgroup.getCounter( Stat.PARFOR_JVMGC_TIME.toString() ) );
				Group cgroup = runjob.getCounters().getGroup(CacheableData.CACHING_COUNTER_GROUP_NAME.toString());
				CacheStatistics.incrementMemHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_MEM.toString() ));
				CacheStatistics.incrementFSBuffHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_FSBUFF.toString() ));
				CacheStatistics.incrementFSHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_FS.toString() ));
				CacheStatistics.incrementHDFSHits((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_HITS_HDFS.toString() ));
				CacheStatistics.incrementFSBuffWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_FSBUFF.toString() ));
				CacheStatistics.incrementFSWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_FS.toString() ));
				CacheStatistics.incrementHDFSWrites((int)cgroup.getCounter( CacheStatistics.Stat.CACHE_WRITES_HDFS.toString() ));
				CacheStatistics.incrementAcquireRTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_ACQR.toString() ));
				CacheStatistics.incrementAcquireMTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_ACQM.toString() ));
				CacheStatistics.incrementReleaseTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_RLS.toString() ));
				CacheStatistics.incrementExportTime(cgroup.getCounter( CacheStatistics.Stat.CACHE_TIME_EXP.toString() ));
			}
				
			// read all files of result variables and prepare for return
			LocalVariableMap[] results = readResultFile(job, resultFile); 

			ret = new RemoteParForJobReturn(runjob.isSuccessful(), 
					                        numTasks, numIters, 
					                        results);  	
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		finally
		{
			// remove created files 
			try
			{
				MapReduceTool.deleteFileIfExistOnHDFS(new Path(resultFile), job);
			}
			catch(IOException ex)
			{
				throw new DMLRuntimeException(ex);
			}
		}
		
		if( DMLScript.STATISTICS ){
			long t1 = System.nanoTime();
			Statistics.maintainCPHeavyHitters("MR-Job_"+jobname, t1-t0);
		}
		
		return ret;
	}
	

	/**
	 * Result file contains hierarchy of workerID-resultvar(incl filename). We deduplicate
	 * on the workerID. Without JVM reuse each task refers to a unique workerID, so we
	 * will not find any duplicates. With JVM reuse, however, each slot refers to a workerID, 
	 * and there are duplicate filenames due to partial aggregation and overwrite of fname 
	 * (the RemoteParWorkerMapper ensures uniqueness of those files independent of the 
	 * runtime implementation). 
	 * 
	 * @param job 
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("deprecation")
	public static LocalVariableMap [] readResultFile( JobConf job, String fname )
		throws DMLRuntimeException, IOException
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		FileSystem fs = FileSystem.get(job);
		Path path = new Path(fname);
		LongWritable key = new LongWritable(); //workerID
		Text value = new Text();               //serialized var header (incl filename)
		
		int countAll = 0;
		for( Path lpath : MatrixReader.getSequenceFilePaths(fs, path) )
		{
			SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(job),lpath,job);
			try
			{
				while( reader.next(key, value) )
				{
					//System.out.println("key="+key.get()+", value="+value.toString());
					if( !tmp.containsKey( key.get() ) )
		        		tmp.put(key.get(), new LocalVariableMap ());	   
					Object[] dat = ProgramConverter.parseDataObject( value.toString() );
		        	tmp.get( key.get() ).put((String)dat[0], (Data)dat[1]);
		        	countAll++;
				}
			}	
			finally
			{
				if( reader != null )
					reader.close();
			}
		}		

		LOG.debug("Num remote worker results (before deduplication): "+countAll);
		LOG.debug("Num remote worker results: "+tmp.size());

		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);	
	}
}
