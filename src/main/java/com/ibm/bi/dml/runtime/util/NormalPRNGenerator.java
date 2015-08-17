/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.util;

import java.util.Random;


/**
 * Class that can generate a stream of random numbers from standard 
 * normal distribution N(0,1). This class internally makes use of 
 * RandNPair, which uses Box-Muller method.
 */


public class NormalPRNGenerator extends PRNGenerator
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	//private long seed;
	Random r;
	private RandNPair pair;
	boolean flag = false; // we use pair.N1 if flag=false, and pair.N2 otherwise
	
	public NormalPRNGenerator() {
		super();
	}
	
	public NormalPRNGenerator(long sd) {
		super();
		setSeed(sd);
	}
	
	/*public NormalPRNGenerator(Random random) {
		init(random);
	}*/
	
	public void setSeed(long sd) {
		//seed = s;
		seed = sd;
		r = new Random(seed);
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
