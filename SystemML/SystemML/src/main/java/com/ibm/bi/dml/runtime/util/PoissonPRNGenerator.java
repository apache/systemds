/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.random.SynchronizedRandomGenerator;
import org.apache.commons.math3.random.Well1024a;


/**
 * Class that can generate a stream of random numbers from Poisson
 * distribution with specified mean. 
 */


public class PoissonPRNGenerator extends PRNGenerator
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	PoissonDistribution _pdist = null;
	double _mean = Double.NaN;
	
	private static final double DEFAULT_MEAN = 1.0;
	
	public PoissonPRNGenerator() {
		// default mean and default seed
		super();
		_mean = 1.0;
		setup(_mean, seed);
	}
	
	public PoissonPRNGenerator(double mean) 
	{
		// default seed
		super();
		_mean = mean;
		setup(_mean, seed);
	}
	
	public PoissonPRNGenerator(long sd) {
		// default mean
		super();
		setup(DEFAULT_MEAN, sd);
	}
	
	public void setup(double mean, long sd) {
		seed = sd;
		
		SynchronizedRandomGenerator srg = new SynchronizedRandomGenerator(new Well1024a());
		srg.setSeed(seed);
		_pdist = new PoissonDistribution(srg, _mean, PoissonDistribution.DEFAULT_EPSILON, PoissonDistribution.DEFAULT_MAX_ITERATIONS);
	}
	
	@Override
	public void setSeed(long sd) {
		setup(_mean, sd);
	}
	
	public void setMean(double mean) {
		setup(mean, seed);
	}
	
	public double nextDouble() {
		return (double) _pdist.sample();
	}
	
	

}
