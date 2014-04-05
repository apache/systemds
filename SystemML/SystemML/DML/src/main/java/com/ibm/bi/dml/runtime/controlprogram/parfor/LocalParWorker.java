/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

/**
 * Instances of this class can be used to execute tasks in parallel. Within each ParWorker 
 * multiple iterations of a single task and subsequent tasks are executed sequentially.
 * 
 * Resiliency approach: retry on computation error, abort on task queue error
 * 
 * 
 */
public class LocalParWorker extends ParWorker implements Runnable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected LocalTaskQueue<Task> _taskQueue   = null;
	
	protected boolean   _stopped     = false;
	protected int 		_max_retry   = -1;
	
	public LocalParWorker( long ID, LocalTaskQueue<Task> q, ParForBody body, int max_retry, boolean monitor )	
	{
		super(ID, body, monitor);

		_taskQueue = q;
		_stopped   = false;
		
		_max_retry = max_retry;
	}
	
	/**
	 * Sets the status to stopped such that execution will be aborted as soon as the
	 * current task is finished.
	 */
	public void setStopped()
	{
		_stopped = true;
	}
	
	@Override
	public void run() 
	{
		// monitoring start
		Timing time1 = ( _monitor ? new Timing(true) : null ); 
		
		// continuous execution (execute tasks until (1) stopped or (2) no more tasks)
		Task lTask = null; 
		
		while( !_stopped ) 
		{
			//dequeue the next task (abort on NO_MORE_TASKS or error)
			try
			{
				lTask = _taskQueue.dequeueTask();
				
				if( lTask == LocalTaskQueue.NO_MORE_TASKS ) // task queue closed (no more tasks)
					break; //normal end of parallel worker
			}
			catch(Exception ex)
			{
				//ex.printStackTrace();
				
				// abort on taskqueue error
				LOG.warn("Error reading from task queue: "+ex.getMessage());
				LOG.warn("Stopping LocalParWorker.");
				break; //no exception thrown to prevent blocking on join
			}
			
			//execute the task sequentially (re-try on error)
			boolean success = false;
			int retrys = _max_retry;
			
			while( !success )
			{
				try 
				{
					///////
					//core execution (see ParWorker)
					executeTask( lTask );
					success = true;
				} 
				catch (Exception ex) 
				{
					LOG.error("Failed to execute "+lTask.toString()+", retry:"+retrys, ex);
					
					if( retrys > 0 )
						retrys--; //retry on task error
					else
					{
						// abort on no remaining retrys
						LOG.error("Error executing task: ",ex);
						LOG.error("Stopping LocalParWorker.");
						break; //no exception thrown to prevent blocking on join 
					}
				}
			}
		}	
		
		//cleanup symbol table
		//cleanupCachedVariables();

		if( _monitor )
		{
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMTASKS, _numTasks);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMITERS, _numIters);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_EXEC_T, time1.stop());
		}
	}

}

	
