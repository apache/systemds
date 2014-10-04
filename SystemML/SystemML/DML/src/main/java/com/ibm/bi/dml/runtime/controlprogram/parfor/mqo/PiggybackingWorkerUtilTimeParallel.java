/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * Extensions: (1) take number of running jobs into account,
 * (2) compute timeout threshold based on max and last job execution time.
 */
public class PiggybackingWorkerUtilTimeParallel extends PiggybackingWorker
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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
				double util = InfrastructureAnalyzer.getClusterUtilization(false);
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
				
				// split job return
				LinkedList<JobReturn> ret = new LinkedList<JobReturn>();
				for( Long id : _minst.ids ){
					ret.add( _minst.constructJobReturn(id, mret) );
					Statistics.decrementNoOfExecutedMRJobs();
				}
				putJobResults(_minst.ids, ret);
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex); 
			}
		}
		
	}
}
