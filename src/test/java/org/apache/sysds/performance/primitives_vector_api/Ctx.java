/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysds.performance.primitives_vector_api;

public class Ctx {
	public int len;
	public double[] a, cInit,b,c, cScalar, cVector;
	public double bval;

	public double scalarRes, vectorRes;
	public double maxDiff;
	public boolean ok;
	public int[] a_int;

	void initDenseA() {
		a = new double[len];
		for (int i = 0; i < len; i++) a[i] = (i % 10) - 5;
	}
	void initDenseB() {
		b = new double[len];
		for (int i = 0; i < len; i++) b[i] = (i % 10) - 5;
	}
	void initDenseC() {
		c = new double[len];
		for (int i = 0; i < len; i++) c[i] = (i % 10) - 5;
	}
	void initDenseAInt() {
		a_int = new int[len];
		for (int i = 0; i < len; i++) a_int[i] = i;;
	}
	
	void initbval(){
		bval = 1.234567;
	}
	void initDenseADiv() {
		a = new double[len];
		for (int i = 0; i < len; i++) {
			a[i] = ((i % 10) + 1);  // Range: 1 to 10 (no zeros)
		}
	}
	void initDenseBDiv() {
		b = new double[len];
		for (int i = 0; i < len; i++) b[i] = ((i % 10) + 1);
	}

	void initDenseAandC_mutable() {
		initDenseADiv();
		cInit = new double[len];
		for (int i = 0; i < len; i++) cInit[i] = (i % 10) - 5;
		cScalar = java.util.Arrays.copyOf(cInit, len);
		cVector = java.util.Arrays.copyOf(cInit, len);
	}

	void resetC() {
		if (cInit != null) {
			System.arraycopy(cInit, 0, cScalar, 0, len);
			System.arraycopy(cInit, 0, cVector, 0, len);
		}
	}
}
