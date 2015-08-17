/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.utils;

public class Timer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	long start;
	double sofar;
	
	public Timer() {
		start = 0 ;
		sofar = 0.0;
	}
	
	public void start() {
		start = System.nanoTime();
		sofar = 0.0;
	}
	
	public double stop() {
		double duration = sofar + (System.nanoTime()-start)*1e-6;
		sofar = 0.0;
		start = 0;
		return duration;
	}
	
	public double nanostop() {
		double duration = sofar + (System.nanoTime()-start);
		sofar = 0.0;
		start = 0;
		return duration;
	}
	
	public double pause() {
		sofar += (System.nanoTime()-start)*1e-6;
		return sofar;
	}
	
	public void resume() {
		start = System.nanoTime();
	}
	
}
