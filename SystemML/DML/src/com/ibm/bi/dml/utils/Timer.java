package com.ibm.bi.dml.utils;

public class Timer {
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
