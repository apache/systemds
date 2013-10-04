/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

/**
 * Helper class for encapsulated time measurements.
 * 
 * 
 */
public class Timing 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _start = -1;
	
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
