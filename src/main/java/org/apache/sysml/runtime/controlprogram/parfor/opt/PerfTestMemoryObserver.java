/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import java.lang.ref.WeakReference;

/**
 * Observer thread for asynchronously monitor the memory consumption.
 * It periodically measures the used memory with period <code>MEASURE_INTERVAL</code>,
 * by explicitly invoking the garbage collector (if required until it is really executed)
 * and afterwards obtaining the currently used memory.
 * 
 * Protocol: (1) measure start, (2) start thread, (3) *do some work*, (4) join thread, (5) get max memory.
 *  
 */
public class PerfTestMemoryObserver implements Runnable
{
	
	public static final int MEASURE_INTERVAL = 50; //in ms 
	
	private long    _startMem = -1;
	private long    _maxMem   = -1; 
	private boolean _stopped  = false;
	
	public PerfTestMemoryObserver()
	{
		_startMem = -1;
		_maxMem   = -1; 
		_stopped  = false;			
	}
		
	/**
	 * 
	 */
	public void measureStartMem()
	{
		forceGC();
		_startMem =  Runtime.getRuntime().totalMemory()
		           - Runtime.getRuntime().freeMemory();
	}
	
	/**
	 * 
	 * @return
	 */
	public long getMaxMemConsumption()
	{
		long val = _maxMem - _startMem;
		return (val < 0) ? 0 : val; 
	}
	
	/**
	 * 
	 */
	public void setStopped()
	{
		_stopped = true;
	}

	@Override
	public void run() 
	{
		try
		{
			while( !_stopped )
			{
				forceGC();
				long value =   Runtime.getRuntime().totalMemory()
		                     - Runtime.getRuntime().freeMemory(); 
				
				_maxMem = Math.max(value, _maxMem);
				
				Thread.sleep( MEASURE_INTERVAL );
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException("Error measuring Java memory usage", ex);
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getUsedMemory()
	{
		forceGC();
		return  ( Runtime.getRuntime().totalMemory()
		           - Runtime.getRuntime().freeMemory() );
	}
	
	/** 
	 * 
	 * @param force
	 */
	private static void forceGC()
	{
		//request gc until weak reference is eliminated by gc
		Object o = new Object();
		WeakReference<Object> ref = new WeakReference<Object>(o); //collected, everytime gc is actually invoked
		while((o=ref.get())!= null) 
			System.gc(); //called on purpose, no production use.
	}
}