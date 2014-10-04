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
public class PiggybackingWorkerUtilDecayParallel extends PiggybackingWorker
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//internal configuration parameters
	private static long MIN_MERGE_INTERVAL = 1000;
	private static double UTILIZATION_DECAY = 0.5; //decay per minute 
	
	//thread pool for parallel submit
	private ExecutorService _parSubmit = null;
	
	private long _minTime = -1;
	private double _utilDecay = -1; 
	private int _par = -1;
	
	public PiggybackingWorkerUtilDecayParallel(int par)
	{
		this( MIN_MERGE_INTERVAL, 
			  UTILIZATION_DECAY, 
			  par );
	}
	
	public PiggybackingWorkerUtilDecayParallel( long minInterval, double utilDecay, int par )
	{
		_minTime = minInterval;
		_utilDecay = utilDecay;
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
				long currentTime = System.currentTimeMillis()+1; //ensure > lastTime
				
				// wait until next submission
				Thread.sleep(_minTime); //wait at least minTime
				double util = InfrastructureAnalyzer.getClusterUtilization(true);
				double utilThreshold = 1-Math.pow(_utilDecay, Math.ceil(((double)currentTime-lastTime)/60000));
				
				//continue to collect jobs if cluster util too high (decay to prevent starvation)
				if( util > utilThreshold ) { //cluster utilization condition
					continue; //1min - >50%, 2min - >75%, 3min - >87.5%, 4min - > 93.7%
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
