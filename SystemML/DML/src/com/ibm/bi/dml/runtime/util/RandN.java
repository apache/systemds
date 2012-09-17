package com.ibm.bi.dml.runtime.util;

import java.util.Random;

/**
 * Class that can generate a stream of random numbers from standard 
 * normal distribution N(0,1). This class internally makes use of 
 * RandNPair, which uses Box-Muller method.
 */


public class RandN {
	//private long seed;
	Random r;
	private RandNPair pair;
	boolean flag = false; // we use pair.N1 if flag=false, and pair.N2 otherwise
	
	public RandN(long s) {
		init(new Random(s));
	}
	
	public RandN(Random random) {
		init(random);
	}
	
	private void init(Random random) {
		//seed = s;
		r = random;
		pair = new RandNPair();
		flag = false;
		pair.compute(r);
	}
	
	public double nextDouble() {
		double d;
		if (!flag) {
			d = pair.getFirst();
		}
		else {
			d = pair.getSecond();
			pair.compute(r);
		}
		flag = !flag;
		return d;
	}

}
