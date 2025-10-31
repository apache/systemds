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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;

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
	
	public static final int    MAX_SIZE      = 100000; //main memory constraint
	public static final Object NO_MORE_TASKS = null; //object to signal NO_MORE_TASKS
	
	private LinkedList<T>  _data        = null;
	private boolean 	   _closedInput = false;
	private DMLRuntimeException _failure = null;
	protected Runnable subscriber = null;
	private static final Log LOG = LogFactory.getLog(LocalTaskQueue.class.getName());
	
	public LocalTaskQueue()
	{
		_data        = new LinkedList<>();
		_closedInput = false;
	}
	
	/**
	 * Synchronized insert of a new task to the end of the FIFO queue.
	 * 
	 * @param t task
	 * @throws InterruptedException if InterruptedException occurs
	 */
	public void enqueueTask( T t )
		throws InterruptedException
	{
		Runnable s;

		synchronized (this) {
			while(_data.size() + 1 > MAX_SIZE && _failure == null) {
				LOG.warn("MAX_SIZE of task queue reached.");
				wait(); //max constraint reached, wait for read
			}

			if(_failure != null)
				throw _failure;

			_data.addLast(t);
			s = subscriber;
			notify();
		}

		if (s != null)
			s.run();
	}
	
	/**
	 * Synchronized read and delete from the top of the FIFO queue.
	 * 
	 * @return task
	 * @throws InterruptedException if InterruptedException occurs
	 */
	@SuppressWarnings("unchecked")
	public synchronized T dequeueTask() 
		throws InterruptedException
	{
		while( _data.isEmpty() && _failure == null )
		{
			if( !_closedInput )
				wait(); // wait for writers
			else
				return (T)NO_MORE_TASKS; 
		}

		if ( _failure != null )
			throw _failure;

		T t = _data.removeFirst();
		
		notify(); // notify waiting writers
		
		return t;
	}

	/**
	 * Synchronized (logical) insert of a NO_MORE_TASKS symbol at the end of the FIFO queue in order to
	 * mark that no more tasks will be inserted into the queue.
	 */
	public void closeInput()
	{
		Runnable s;

		synchronized (this) {
			if(_closedInput)
				return;

			_closedInput = true;
			s = subscriber;
			notifyAll(); // Notify all the waiting readers
		}

		if (s != null)
			s.run();

		subscriber = null;
	}
	
	public synchronized boolean isProcessed() {
		return _closedInput && _data.isEmpty();
	}

	public synchronized void propagateFailure(DMLRuntimeException failure) {
		if (_failure == null) {
			_failure = failure;
			notifyAll();
		}
	}

	public void setSubscriber(Runnable subscriber) {
		int size;
		boolean closed;
		synchronized (this) {
			if (this.subscriber != null)
				throw new DMLRuntimeException("Cannot set multiple subscribers.");

			this.subscriber = subscriber;
			size = _data.size();
			closed = _closedInput;
		}
		for (int i = 0; i < size; i++)
			subscriber.run();
		if (closed)
			subscriber.run();
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
