/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.util.Random;


public class UniformPRNGenerator extends PRNGenerator {

	Random runif = null;
	
	public void init(long sd) {
		seed = sd;
		runif = new Random(seed);
	}
	
	public UniformPRNGenerator(long sd) {
		super();
		init(sd);
	}

	public UniformPRNGenerator() {
		super();
	}

	@Override
	public double nextDouble() {
		return runif.nextDouble();
	}

	public int nextInt(int n) {
		return runif.nextInt(n);
	}
}
