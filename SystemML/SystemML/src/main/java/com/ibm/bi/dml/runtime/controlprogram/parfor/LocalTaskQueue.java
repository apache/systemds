/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * This class provides a way of dynamic task distribution to multiple workers
 * in local multi-threaded environments. Its main purpose is inter-thread communication
 * for achieving dynamic load balancing. A good load balance between parallel workers is crucial
 * with regard to the overall speedup of parallelization (see Amdahl's law).
 * 
 * From a technical perspective, a thread monitor concept is used for blocking of waiting writers
 * and readers. Synchronized writes and reads ensure that each task is only read by exactly one
 * reader. Furthermore, the queue is implemented as a simple FIFO.
 * 
 *
 */
public class LocalTaskQueue<T> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final int    MAX_SIZE      = 100000; //main memory constraint
	public static final Object NO_MORE_TASKS = null; //object to signal NO_MORE_TASKS
	
	private LinkedList<T>  _data        = null;
	private boolean 	   _closedInput = false; 
	private static final Log LOG = LogFactory.getLog(LocalTaskQueue.class.getName());
	
	public LocalTaskQueue()
	{
		_data        = new LinkedList<T>();
		_closedInput = false;
	}
	
	/**
	 * Synchronized insert of a new task to the end of the FIFO queue.
	 * 
	 * @param t
	 * @throws InterruptedException
	 */
	public synchronized void enqueueTask( T t ) 
		throws InterruptedException
	{
		while( _data.size() + 1 > MAX_SIZE )
		{
			LOG.warn("MAX_SIZE of task queue reached.");
			wait(); //max constraint reached, wait for read
		}
		
		_data.addLast( t );
		
		notify(); //notify waiting readers
	}
	
	/**
	 * Synchronized read and delete from the top of the FIFO queue.
	 * 
	 * @return
	 * @throws InterruptedException
	 */
	@SuppressWarnings("unchecked")
	public synchronized T dequeueTask() 
		throws InterruptedException
	{
		while( _data.isEmpty() )
		{
			if( !_closedInput )
				wait(); // wait for writers
			else
				return (T)NO_MORE_TASKS; 
		}
		
		T t = _data.removeFirst();
		
		notify(); // notify waiting writers
		
		return t;
	}
	
	/**
	 * Synchronized (logical) insert of a NO_MORE_TASKS symbol at the end of the FIFO queue in order to
	 * mark that no more tasks will be inserted into the queue.
	 */
	public synchronized void closeInput()
	{
		_closedInput = true;
		notifyAll(); //notify all waiting readers
	}
	
	/**
	 * Synchronized read of the current number of tasks in the queue.
	 * 
	 * @return
	 * @throws InterruptedException
	 */
	public synchronized int size()
		throws InterruptedException
	{
		return _data.size();
	}

	@Override
	public synchronized String toString() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append("TASK QUEUE (size=");
		sb.append(_data.size());
		sb.append(",close=");
		sb.append(_closedInput);
		sb.append(")\n");
		
		int count = 1;
		for( T t : _data )
		{
			sb.append("  TASK #");
			sb.append(count);
			sb.append(": ");
			sb.append(t.toString());
			sb.append("\n");
			
			count++;
		}
		
		return sb.toString();
	}
}
