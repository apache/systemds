/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.util.Random;

/**
 * Class that generates a pair of random numbers from standard normal 
 * distribution N(0,1). Box-Muller method is used to compute random 
 * numbers from N(0,1) using two independent random numbers from U[0,1).
 */

public class RandNPair 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
