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

package org.apache.sysml.runtime.util;

import java.util.Random;


public class UniformPRNGenerator extends PRNGenerator {

	Random runif = null;
	
	public void setSeed(long sd) {
		seed = sd;
		runif = new Random(seed);
	}
	
	public UniformPRNGenerator(long sd) {
		super();
		setSeed(sd);
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
