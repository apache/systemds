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

package org.apache.sysml.runtime.controlprogram.parfor.stat;

/**
 * Helper class for encapsulated time measurements.
 * 
 * 
 */
public class Timing 
{

	
	private long _start = -1;
	
	public Timing() {
		//default constructor
	}
	
	public Timing(boolean start)
	{
		//init and start the timer
		if( start ){
			start();
		}
	}
	
	/**
	 * Starts the time measurement.
	 */
	public void start()
	{
		_start = System.nanoTime();
	}
	
	/**
	 * Measures and returns the time since the last start() or stop() invocation and
	 * restarts the measurement.
	 * 
	 * @return
	 */
	public double stop()
	{
		if( _start == -1 )
			throw new RuntimeException("Stop time measurement without prior start is invalid.");
	
		long end = System.nanoTime();		
		double duration = ((double)(end-_start))/1000000;
		
		//carry end time over
		_start = end;		
		return duration;
	}
	
	/**
	 * Measures and returns the time since the last start() or stop() invocation,
	 * restarts the measurement, and prints the last measurement to STDOUT.
	 */
	public void stopAndPrint()
	{
		double tmp = stop();
		
		System.out.println("PARFOR: time = "+tmp+"ms");
	}
	
}
