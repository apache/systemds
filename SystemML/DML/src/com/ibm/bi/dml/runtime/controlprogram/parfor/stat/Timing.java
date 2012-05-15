package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

/**
 * Helper class for encapsulated time measurements.
 * 
 * 
 * @author mboehm
 */
public class Timing 
{
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
			throw new RuntimeException("Time measurement sotp without prior start not applicable.");
	
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
		
		System.out.println("PARFOR: generic time measurement >>> "+tmp+"ms.");
	}
	
}
