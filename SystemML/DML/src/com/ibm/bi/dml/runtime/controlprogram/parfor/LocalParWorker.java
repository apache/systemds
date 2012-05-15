package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
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
 * @author mboehm
 */
public class LocalParWorker extends ParWorker implements Runnable
{
	protected LocalTaskQueue _taskQueue   = null;
	protected boolean        _stopped     = false;
	
	protected int 			 _max_retry   = -1;
	
	public LocalParWorker( long ID, LocalTaskQueue q, ParForBody body, int max_retry )	
	{
		super(ID, body);

		_taskQueue   = q;
		_stopped     = false;
		
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
		Timing time1;
		if( ParForProgramBlock.MONITOR )
		{
			time1 = new Timing(); 
			time1.start();
		}
		
		// continuous execution:
		// execute tasks until (1) stopped or (2) no more tasks
		while( !_stopped ) 
		{
			//dequeue the next task (abort on NO_MORE_TASKS or error)
			Task lTask = null; 
			try
			{
				lTask = _taskQueue.dequeueTask();
				
				if( lTask == LocalTaskQueue.NO_MORE_TASKS ) // task queue closed (no more tasks)
					break; //normal end of parallel worker
			}
			catch(Exception ex)
			{
				// abort on taskqueue error
				throw new RuntimeException(ex); 
			}
			
			//execute the task sequentially (re-try on error)
			boolean success = false;
			int retrys = _max_retry;
			
			while( !success )
			{
				try 
				{
					//core execution (see ParWorker)
					executeTask( lTask );
					success = true;
				} 
				catch (Exception ex) 
				{
					System.out.println("ParFOR: Failed to execute task "+lTask.toString()+" retry:"+retrys);
					
					if( retrys > 0 )
						retrys--; //retry on task error
					else
						throw new RuntimeException(ex); 
				}
			}
		}	
		
		if( ParForProgramBlock.MONITOR )
		{
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMTASKS, _numTasks);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMITERS, _numIters);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_EXEC_T, time1.stop());
		}
	}

	
	/* 
	@Override
	public void run() 
	{
		// monitoring start
		Timing time1;
		if( ParForProgramBlock.MONITOR )
		{
			time1 = new Timing(); 
			time1.start();
		}
		
		// continuous execution:
		// execute tasks until (1) stopped or (2) no more tasks
		try
		{
			//dequeue the next task and execute
			Task lTask = null; 
			while( (lTask = _taskQueue.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS && !_stopped ) 
			{
				//execute the task 
				executeTask( lTask );
			}
		}	
		catch(Exception ex)	
		{
			throw new RuntimeException("ParFOR: Failed to execute task",ex); 
		}	
		
		if( ParForProgramBlock.MONITOR )
		{
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMTASKS, _numTasks);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_NUMITERS, _numIters);
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_EXEC_T, time1.stop());
		}
	}
	*/
}

	
