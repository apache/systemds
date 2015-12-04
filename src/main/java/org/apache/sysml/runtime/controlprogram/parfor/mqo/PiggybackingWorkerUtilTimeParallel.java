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

package org.apache.sysml.runtime.controlprogram.parfor.mqo;

import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.sysml.lops.runtime.RunMRJobs;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.Statistics;

/**
 * 
 * Extensions: (1) take number of running jobs into account,
 * (2) compute timeout threshold based on max and last job execution time.
 */
public class PiggybackingWorkerUtilTimeParallel extends PiggybackingWorker
{

	//internal configuration parameters
	private static long MIN_MERGE_INTERVAL = 1000;
	private static long MAX_MERGE_INTERVAL = 60000;
	private static double UTILIZATION_THRESHOLD = 0.4; //60% occupied map tasks
	
	//thread pool for parallel submit
	private ExecutorService _parSubmit = null;
	
	private long _minTime = -1;
	private long _maxTime = -1;
	private double _utilThreshold = -1; 
	private int _par = -1;
	
	public PiggybackingWorkerUtilTimeParallel(int par)
	{
		this( MIN_MERGE_INTERVAL, 
			  MAX_MERGE_INTERVAL, 
			  UTILIZATION_THRESHOLD, 
			  par );
	}
	
	public PiggybackingWorkerUtilTimeParallel( long minInterval, long maxInterval, double utilThreshold, int par )
	{
		_minTime = minInterval;
		_maxTime = maxInterval;
		_utilThreshold = utilThreshold;
		_par = par;
		
		//init thread pool
		_parSubmit = Executors.newFixedThreadPool(_par);
	}

	@Override 
	public void setStopped()
	{
		//parent logic
		super.setStopped();
		
		//explicitly stop the thread pool
		_parSubmit.shutdown();
	}
	
	@Override
	public void run() 
	{
		long lastTime = System.currentTimeMillis();
		
		while( !_stop )
		{
			try
			{
				long currentTime = System.currentTimeMillis();
				
				// wait until next submission
				Thread.sleep(_minTime); //wait at least minTime
				if( RuntimePiggybacking.isEmptyJobPool() )
					continue;
				double util = RuntimePiggybackingUtils.getCurrentClusterUtilization();
				if(   util > _utilThreshold           //cluster utilization condition
				   && currentTime-lastTime<_maxTime ) //timeout condition 
				{
					continue;
				}
				
				// pick job type with largest number of jobs
				LinkedList<Pair<Long,MRJobInstruction>> workingSet = RuntimePiggybacking.getMaxWorkingSet();
				if( workingSet == null )
					continue; //empty pool
				
				// merge jobs (if possible)
				LinkedList<MergedMRJobInstruction> mergedWorkingSet = mergeMRJobInstructions(workingSet);
				
				// submit all resulting jobs (parallel submission)
				for( MergedMRJobInstruction minst : mergedWorkingSet )
				{
					//submit job and return results if finished
					_parSubmit.execute(new MRJobSubmitTask(minst));
				}
				
				lastTime = currentTime;
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex);
			}
		}
	}
	
	
	/**
	 * 
	 * 
	 */
	public class MRJobSubmitTask implements Runnable
	{
		private MergedMRJobInstruction _minst = null;
		
		public MRJobSubmitTask( MergedMRJobInstruction minst )
		{
			_minst = minst;
		}
		
		@Override
		public void run() 
		{
			try
			{
				// submit mr job
				JobReturn mret = RunMRJobs.submitJob(_minst.inst);
				Statistics.incrementNoOfExecutedMRJobs();

				// error handling
				if( !mret.successful )
					LOG.error("Failed to run merged mr-job instruction:\n"+_minst.inst.toString()); 
				
				// split job return
				LinkedList<JobReturn> ret = new LinkedList<JobReturn>();
				for( Long id : _minst.ids ){
					ret.add( _minst.constructJobReturn(id, mret) );
					Statistics.decrementNoOfExecutedMRJobs();
				}
				// make job returns available and notify waiting clients
				putJobResults(_minst.ids, ret);
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex); 
			}
		}
		
	}
}
