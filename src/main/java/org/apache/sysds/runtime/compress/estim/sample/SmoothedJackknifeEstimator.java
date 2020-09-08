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

package org.apache.sysds.runtime.compress.estim.sample;

import org.apache.sysds.runtime.compress.utils.Bitmap;

public class SmoothedJackknifeEstimator {

	/**
	 * Peter J. Haas, Jeffrey F. Naughton, S. Seshadri, and Lynne Stokes. Sampling-Based Estimation of the Number of
	 * Distinct Values of an Attribute. VLDB'95, Section 4.3.
	 * 
	 * @param ubm        The Uncompressed Bitmap containing the data from the sample
	 * @param nRows      The original number of rows in the entire input
	 * @param sampleSize The number of rows in the sample
	 * @return Estimate of the number of distinct values
	 */
	public static int get(Bitmap ubm, int nRows, int sampleSize) {
		int numVals = ubm.getNumValues();
		int[] freqCounts = FrequencyCount.get(ubm);
		// all values in the sample are zeros
		if(freqCounts.length == 0)
			return 0;
		// nRows is N and sampleSize is n

		int d = numVals;
		double f1 = freqCounts[0];
		int Nn = nRows * sampleSize;
		double D0 = (d - f1 / sampleSize) / (1 - (nRows - sampleSize + 1) * f1 / Nn);
		double NTilde = nRows / D0;
		/*-
		 *
		 * h (as defined in eq. 5 in the paper) can be implemented as:
		 * 
		 * double h = Gamma(nRows - NTilde + 1) x Gamma.gamma(nRows -sampleSize + 1) 
		 * 		     ----------------------------------------------------------------
		 *  		Gamma.gamma(nRows - sampleSize - NTilde + 1) x Gamma.gamma(nRows + 1)
		 * 
		 * 
		 * However, for large values of nRows, Gamma.gamma returns NAN
		 * (factorial of a very large number).
		 * 
		 * The following implementation solves this problem by levaraging the
		 * cancelations that show up when expanding the factorials in the
		 * numerator and the denominator.
		 * 
		 * 
		 * 		min(A,D-1) x [min(A,D-1) -1] x .... x B
		 * h = -------------------------------------------
		 * 		C x [C-1] x .... x max(A+1,D)
		 * 
		 * where A = N-\tilde{N}
		 *       B = N-\tilde{N} - n + a
		 *       C = N
		 *       D = N-n+1
		 *       
		 * 		
		 *
		 */
		double A = nRows - NTilde;
		double B = A - sampleSize + 1;
		double C = nRows;
		double D = nRows - sampleSize + 1;
		A = Math.min(A, D - 1);
		D = Math.max(A + 1, D);
		double h = 1;

		for(; A >= B || C >= D; A--, C--) {
			if(A >= B)
				h *= A;
			if(C >= D)
				h /= C;
		}
		// end of h computation

		double g = 0, gamma = 0;
		// k here corresponds to k+1 in the paper (the +1 comes from replacing n
		// with n-1)
		for(int k = 2; k <= sampleSize + 1; k++) {
			g += 1.0 / (nRows - NTilde - sampleSize + k);
		}
		for(int i = 1; i <= freqCounts.length; i++) {
			gamma += i * (i - 1) * freqCounts[i - 1];
		}
		gamma *= (nRows - 1) * D0 / Nn / (sampleSize - 1);
		gamma += D0 / nRows - 1;

		double estimate = (d + nRows * h * g * gamma) / (1 - (nRows - NTilde - sampleSize + 1) * f1 / Nn);
		return estimate < 1 ? 1 : (int) Math.round(estimate);
	}
}
