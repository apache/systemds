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
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator.CountDistinctTypes;
import org.apache.sysds.utils.Hash;
import org.apache.sysds.utils.Hash.HashType;

/**
 * This class contains estimation operations for matrix block.
 */
public class LibMatrixCountDistinct {

	// ------------------------------
	// Logging parameters:
	// local debug flag
	private static final boolean LOCAL_DEBUG = true;
	// DEBUG/TRACE for details
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;

	private static final Log LOG = LogFactory.getLog(LibMatrixCountDistinct.class.getName());

	static {
		// for internal debugging only
		if (LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.matrix.data").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}
	// ------------------------------

	static public int minimumSize = 64000;

	private LibMatrixCountDistinct() {
		// Prevent instantiation via private constructor.
	}

	/**
	 * Public method to count the number of distinct values inside a matrix.
	 * Depending on which CountDistinctOperator selected it either gets the absolute
	 * number or a estimated value.
	 * 
	 * TODO: support counting num distinct in rows, or columns axis.
	 * 
	 * TODO: If the MatrixBlock type is CompressedMatrix, simply read the vaules from the ColGroups.
	 * 
	 * @param in the input matrix to count number distinct values in
	 * @param op the selected operator to use
	 * @return the distinct count
	 */
	public static int estimateDistinctValues(MatrixBlock in, CountDistinctOperator op) {
		// set output to correct size.

		int res = 0;
		if (op.hashType == HashType.ExpHash && op.operatorType == CountDistinctTypes.KMV) {
			throw new DMLException("Invalid hashing configuration using " + HashType.ExpHash + " and " + CountDistinctTypes.KMV);
		}

		// shortcut in simplest case.
		if(in.getNumColumns() == 1 && in.getNumRows() == 1){
			return 1;
		}

		// Just use naive implementation if the size is small.
		if (in.getNumRows() * in.getNumColumns() < minimumSize) {
			res = CountDistinctValuesNaive(in);
		} else {
			switch (op.operatorType) {
				case COUNT:
					res = CountDistinctValuesNaive(in);
					break;
				case KMV:
					res = CountDistinctValuesKVM(in, op);
					break;
				case HLL:
					throw new NotImplementedException("HyperLogLog not implemented");
					// res = CountDistinctHyperLogLog(in);
					// break;
				default:
					throw new DMLException("Invalid or not implemented Estimator Type");
			}
		}

		if (res == 0)
			throw new DMLRuntimeException("Imposible estimate of distinct values");
		return res;
	}

	/**
	 * Naive implementation of counting Distinct values. Benefit Precise, but Uses
	 * memory, on the scale of input.
	 * 
	 * @param in the input matrix to count number distinct values in
	 * @return the distinct count
	 */
	private static int CountDistinctValuesNaive(MatrixBlock in) {
		Set<Double> distinct = new HashSet<Double>();
		if(in.isInSparseFormat()){
			Iterator<IJV> it = in.getSparseBlockIterator();
			while(it.hasNext()){
				distinct.add(it.next().getV());
			}
		} else{
			double[] data = in.getDenseBlockValues();
			if (data == null){
				throw new DMLRuntimeException("Not valid execution");
			}
			for (double v : data){
				distinct.add(v);
			}
		}
		return distinct.size();
	}

	/**
	 * KMV synopsis(for k minimum values) Distinct-Value Estimation
	 * 
	 * Kevin S. Beyer, Peter J. Haas, Berthold Reinwald, Yannis Sismanis, Rainer
	 * Gemulla: On synopses for distinct‐value estimation under multiset operations.
	 * SIGMOD 2007
	 * 
	 * @param in
	 * @return the distinct count
	 */
	private static int CountDistinctValuesKVM(MatrixBlock in, CountDistinctOperator op) {

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

		for (int c = 0; c < in.getNumColumns(); c++) {
			for (int r = 0; r < in.getNumRows(); r++) {
				double input = in.getValue(r, c);
				// >>>1 removes the signing bit from the has value, such that the value no
				// longer is negative.
				int v = (Math.abs(Hash.hash(new Double(input), op.hashType))) % (M - 1) + 1;

				if (!cache.contains(v)) {
					if (smallestHashes.size() < k) {
						smallestHashes.add(v);
						cache.add(v);
					} else if (v < smallestHashes.peek()) {
						smallestHashes.add(v);
						cache.remove(smallestHashes.poll());
					}
				}

			}
		}

		LOG.debug("M: " + M);
		LOG.debug("smallest hash:" + smallestHashes.peek());

		if (smallestHashes.size() < k) {
			return smallestHashes.size();
		} else {
			// LOG.debug("Priority Q: ");
			// LOG.debug(Arrays.toString(smallestHashes.toArray()));
			double U_k = (double) smallestHashes.poll() / (double) M;
			LOG.debug("U_k : " + U_k);
			double estimate = (double) (k - 1) / U_k;
			LOG.debug("Estimate: " + estimate);
			double ceilEstimate = Math.min(estimate, (double) D);
			LOG.debug("ceil worst case: " + ceilEstimate);
			// Bounded by maximum number of cells D.
			return (int) ceilEstimate;
		}
	}

	private static int CountDistinctHyperLogLog(MatrixBlock in) {
		return 0;
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