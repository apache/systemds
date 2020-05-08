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

package org.apache.sysds.runtime.matrix.data;

import java.util.Collections;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.runtime.matrix.operators.EstimatorOperator;
import org.apache.sysds.utils.Hash;
import org.apache.sysds.utils.Hash.HashType;

/**
 * This class contains estimation operations for matrix block.
 */
public class LibMatrixEstimator {

	// ------------------------------
	// Logging parameters:
	// local debug flag
	private static final boolean LOCAL_DEBUG = true;
	// DEBUG/TRACE for details
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;

	private static final Log LOG = LogFactory.getLog(LibMatrixEstimator.class.getName());

	static {
		// for internal debugging only
		if(LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.matrix.data").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}
	// ------------------------------

	public enum EstimatorType {
		NUM_DISTINCT_COUNT, // Baseline naive implementation, iterate though, add to hashMap.
		NUM_DISTINCT_KMV, // K-Minimum Values algorithm.
		NUM_DISTINCT_HYPER_LOG_LOG // HyperLogLog algorithm.
	}

	static public int minimumSize = 64000;

	private LibMatrixEstimator() {
		// Prevent instantiation via private constructor.
	}

	/**
	 * Public method to count the number of distinct values inside a matrix. Depending on which EstimatorOperator
	 * selected it either gets the absolute number or a estimated value.
	 * 
	 * TODO: support counting num distinct in rows, or columns axis.
	 * 
	 * @param in  the input matrix to count number distinct values in
	 * @param out the output matrix containing the count dims 1x1
	 * @param op  the selected operator to use
	 */
	public static void EstimateDistinctValues(MatrixBlock in, MatrixBlock out, EstimatorOperator op) {
		// set output to correct size.
		if(out.getNumRows() != 1 && out.getNumColumns() != 1) {
			out = new MatrixBlock(1, 1, 1.0);
		}

		// TODO: If the MatrixBlock type is CompressedMatrix, simply read the vaules from the ColGroups.

		if(op.hashType == HashType.ExpHash && op.operatorType == EstimatorType.NUM_DISTINCT_KMV) {
			throw new DMLException(
				"Invalid hashing configuration using " + HashType.ExpHash + " and " + EstimatorType.NUM_DISTINCT_KMV);
		}

		if(op.operatorType == EstimatorType.NUM_DISTINCT_HYPER_LOG_LOG)
			throw new NotImplementedException("HyperLogLog not implemented");

		// Just use naive implementation if the size is small.
		if(in.getNumRows() * in.getNumColumns() < minimumSize) {
			CountDistinctValuesNaive(in, out);
			return;
		}

		switch(op.operatorType) {
			case NUM_DISTINCT_COUNT:
				CountDistinctValuesNaive(in, out);
				break;
			case NUM_DISTINCT_KMV:
				CountDistinctValuesKVM(in, out, op);
				break;
			case NUM_DISTINCT_HYPER_LOG_LOG:
				CountDistinctHyperLogLog(in, out);
				break;
			default:
				throw new DMLException("Invalid or not implemented Estimator Type");
		}
	}

	/**
	 * Naive implementation of counting Distinct values. Benefit Precise, but Uses memory, on the scale of input.
	 * 
	 * @param in  the input matrix to count number distinct values in
	 * @param out the output matrix of dims 1x1
	 */
	private static void CountDistinctValuesNaive(MatrixBlock in, MatrixBlock out) {
		// Make a hash set to contain all the distinct.
		// Memory usage scale linear with number of distinct values.

		Set<Double> distinct = new HashSet<Double>();

		for(int c = 0; c < in.getNumColumns(); c++) {
			for(int r = 0; r < in.getNumRows(); r++) {
				distinct.add(in.getValue(r, c));
			}
		}

		out.quickSetValue(0, 0, distinct.size());

		// debugging
		// LOG.debug("Distinct actual array: " + Arrays.toString(distinct.toArray()));
		// LOG.debug("Number destinct: " + out);

	}

	/**
	 * KMV synopsis(for k minimum values) Distinct-Value Estimation
	 * 
	 * Kevin S. Beyer, Peter J. Haas, Berthold Reinwald, Yannis Sismanis, Rainer Gemulla: On synopses for distinct‐value
	 * estimation under multiset operations. SIGMOD 2007
	 * 
	 * @param in
	 * @param out
	 */
	private static void CountDistinctValuesKVM(MatrixBlock in, MatrixBlock out, EstimatorOperator op) {

		// D is the number of possible distinct values in the MatrixBlock.
		// As a default we set this number to numRows * numCols
		int D = in.getNumRows() * in.getNumColumns();

		// To ensure that the likelihood to hash to the same value we need O(D^2) bits
		// to hash to assign.
		// This is handled by our custom bit array class.
		long tmp = (long) D * (long) D;
		LOG.debug("M not forced to int size: " + tmp);
		int M = (tmp > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) tmp;

		// the estimator is asymptotically unbiased as k becomes large
		// memory scales with k.
		// D >> k >> 0
		int k = 1024;

		PriorityQueue<Integer> smallestHashes = new PriorityQueue<>(k, Collections.reverseOrder());

		// cache for common elements.
		Set<Integer> cache = new HashSet<>();
		// int maxCache = 512;

		// int gccCount = 0;

		for(int c = 0; c < in.getNumColumns(); c++) {
			for(int r = 0; r < in.getNumRows(); r++) {
				double input = in.getValue(r, c);
				// >>>1 removes the signing bit from the has value, such that the value no longer is negative.
				int v = (Math.abs(Hash.hash(new Double(input), op.hashType))) % (M - 1) + 1;

				if(!cache.contains(v)) {
					if(smallestHashes.size() < k) {
						smallestHashes.add(v);
						cache.add(v);
					}
					else if(v < smallestHashes.peek()) {
						smallestHashes.add(v);
						cache.remove(smallestHashes.poll());
					}
				}

			}
		}

		LOG.debug("M: " + M);
		LOG.debug("smallest hash:" + smallestHashes.peek());

		if(smallestHashes.size() < k) {
			out.quickSetValue(0, 0, smallestHashes.size());
		}
		else {
			// LOG.debug("Priority Q: ");
			// LOG.debug(Arrays.toString(smallestHashes.toArray()));
			double U_k = (double) smallestHashes.poll() / (double) M;
			LOG.debug("U_k : " + U_k);
			double estimate = (double) (k - 1) / U_k;
			LOG.debug("Estimate: " + estimate);
			double ceilEstimate = Math.min(estimate, (double) D);
			LOG.debug("ceil worst case: " + ceilEstimate);
			// Bounded by maximum number of cells D.
			out.quickSetValue(0, 0, ceilEstimate);
		}
	}

	private static void CountDistinctHyperLogLog(MatrixBlock in, MatrixBlock out) {

		// int logm = 2;
		// int m = 1 << logm; // 2 ^ logm
		// byte[] M = new byte[m];

		// for(int c = 0; c < in.getNumColumns(); c++) {
		// for(int r = 0; r < in.getNumRows(); r++) {
		// int xh = new Double(in.getValue(r, c)).hashCode();

		// int i = Hash.linearHash(xh, logm);
		// byte val = (byte) Hash.expHash(xh);

		// if(val > M[i])
		// M[i] = val;

		// }
		// }

		// double wsum = 0;
		// int zerosum = 0;
		// for(int j = 0; j < m; j++) {
		// wsum += Math.pow(2.0, -M[j]);
		// if(M[j] == 0)
		// zerosum++;
		// }
		// double Z = 1 / wsum;
		// double estimate = m * m * Z * 0.7213 / (1 + 1.079 / m);
		// if((estimate < 2.5 * m) && (zerosum > 0))
		// estimate = m * Math.log((double) m / zerosum);

		// LOG.debug("Estimate: " + estimate);
		// // Bounded by maximum number of cells D.
		// out.quickSetValue(0, 0, estimate);

	}
}