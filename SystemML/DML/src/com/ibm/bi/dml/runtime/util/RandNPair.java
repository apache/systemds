package com.ibm.bi.dml.runtime.util;

import java.util.Random;

/**
 * Class that generates a pair of random numbers from standard normal 
 * distribution N(0,1). Box-Muller method is used to compute random 
 * numbers from N(0,1) using two independent random numbers from U[0,1).
 */

public class RandNPair {
	private double N1, N2;
	
	public double getFirst() { return N1; }
	public double getSecond() { return N2; }
	
	public void compute(double U1, double U2) {
		// Box-Muller transform
		double v1 = Math.sqrt(-2*Math.log(U1));
		double v2 = 2*Math.PI*U2;
		N1 = v1 * Math.cos(v2);
		N2 = v1 * Math.sin(v2);
	}
	
	public void compute(Random r) {
		compute(r.nextDouble(), r.nextDouble());
	}
	
	public void compute(long seed) {
		compute(new Random(seed));
	}

}
