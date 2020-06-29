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
 * This class contains various methods for counting the number of distinct values inside a MatrixBlock
 */
public class LibMatrixCountDistinct {

	// ------------------------------
	// Logging parameters:
	// local debug flag
	private static final boolean LOCAL_DEBUG = false;
	// DEBUG/TRACE for details
	private static final Level LOCAL_DEBUG_LEVEL = Level.DEBUG;

	private static final Log LOG = LogFactory.getLog(LibMatrixCountDistinct.class.getName());

	static {
		// for internal debugging only
		if(LOCAL_DEBUG) {
			Logger.getLogger("org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct").setLevel(LOCAL_DEBUG_LEVEL);
		}
	}
	// ------------------------------

	/**
	 * The minimum number NonZero of cells in the input before using approximate techniques for counting number of
	 * distinct values.
	 */
	public static int minimumSize = 1024;

	private LibMatrixCountDistinct() {
		// Prevent instantiation via private constructor.
	}

	/**
	 * Public method to count the number of distinct values inside a matrix. Depending on which CountDistinctOperator
	 * selected it either gets the absolute number or a estimated value.
	 * 
	 * TODO: Support counting num distinct in rows, or columns axis.
	 * 
	 * TODO: Add support for distributed spark operations
	 * 
	 * TODO: If the MatrixBlock type is CompressedMatrix, simply read the vaules from the ColGroups.
	 * 
	 * @param in the input matrix to count number distinct values in
	 * @param op the selected operator to use
	 * @return the distinct count
	 */
	public static int estimateDistinctValues(MatrixBlock in, CountDistinctOperator op) {
		int res = 0;
		if(op.operatorType == CountDistinctTypes.KMV &&
			(op.hashType == HashType.ExpHash || op.hashType == HashType.StandardJava)) {
			throw new DMLException("Invalid hashing configuration using " + op.hashType + " and " + op.operatorType);
		}
		else if(op.operatorType == CountDistinctTypes.HLL) {
			throw new NotImplementedException("HyperLogLog not implemented");
		}
		// shortcut in simplest case.
		if( in.getLength() == 1 || in.isEmpty() )
			return 1;
		else if( in.getNonZeros() < minimumSize ) {
			// Just use naive implementation if the number of nonZeros values size is small.
			res = countDistinctValuesNaive(in);
		}
		else {
			switch(op.operatorType) {
				case COUNT:
					res = countDistinctValuesNaive(in);
					break;
				case KMV:
					res = countDistinctValuesKVM(in, op);
					break;
				default:
					throw new DMLException("Invalid or not implemented Estimator Type");
			}
		}
		
		if(res == 0)
			throw new DMLRuntimeException("Imposible estimate of distinct values");
		return res;
	}

	/**
	 * Naive implementation of counting Distinct values.
	 * 
	 * Benefit Precise, but uses memory, on the scale of inputs number of distinct values.
	 * 
	 * @param in The input matrix to count number distinct values in
	 * @return The absolute distinct count
	 */
	private static int countDistinctValuesNaive(MatrixBlock in) {
		Set<Double> distinct = new HashSet<>();

		// TODO performance: direct sparse block /dense block access
		if(in.isInSparseFormat()) {
			Iterator<IJV> it = in.getSparseBlockIterator();
			while(it.hasNext()) {
				distinct.add(it.next().getV());
			}
			if( in.getNonZeros() < in.getLength() )
				distinct.add(0d);
		}
		else {
			//TODO fix for large dense blocks, where this call will fail
			double[] data = in.getDenseBlockValues();
			if(data == null) {
				throw new DMLRuntimeException("Not valid execution");
			}
			//TODO avoid redundantly adding zero if not entirly dense
			for(double v : data) {
				distinct.add(v);
			}
		}
		return distinct.size();
	}

	/**
	 * KMV synopsis(for k minimum values) Distinct-Value Estimation
	 * 
	 * Kevin S. Beyer, Peter J. Haas, Berthold Reinwald, Yannis Sismanis, Rainer Gemulla:
	 * 
	 * On synopses for distinct‐value estimation under multiset operations. SIGMOD 2007
	 * 
	 * TODO: Add multi-threaded version
	 * 
	 * @param in The Matrix Block to estimate the number of distinct values in
	 * @return The distinct count estimate
	 */
	private static int countDistinctValuesKVM(MatrixBlock in, CountDistinctOperator op) {

		// D is the number of possible distinct values in the MatrixBlock.
		// plus 1 to take account of 0 input.
		long D = in.getNonZeros() + 1;

		/**
		 * To ensure that the likelihood to hash to the same value we need O(D^2) positions to hash to assign. If the
		 * value is higher than int (which is the area we hash to) then use Integer Max value as largest hashing space.
		 */
		long tmp = D * D;
		int M = (tmp > (long) Integer.MAX_VALUE) ? Integer.MAX_VALUE : (int) tmp;
		LOG.debug("M not forced to int size: " + tmp);
		LOG.debug("M: " + M);
		/**
		 * The estimator is asymptotically unbiased as k becomes large, but memory usage also scales with k. Furthermore
		 * k value must be within range: D >> k >> 0
		 */
		int k = D > 64 ? 64 : (int) D;
		SmallestPriorityQueue spq = new SmallestPriorityQueue(k);

		if(in.isInSparseFormat()) {
			Iterator<IJV> it = in.getSparseBlockIterator();
			while(it.hasNext()) {
				double fullValue = it.next().getV();
				int hash = Hash.hash(fullValue, op.hashType);
				// Since Java does not have unsigned integer, the hash value is abs.
				int v = (Math.abs(hash)) % (M - 1) + 1;
				spq.add(v);
			}
			if( in.getNonZeros() < in.getLength() )
				spq.add(Hash.hash(0d, op.hashType));
		}
		else {
			//TODO fix for large dense blocks, where this call will fail
			double[] data = in.getDenseBlockValues();
			for(double fullValue : data) {
				int hash = Hash.hash(fullValue, op.hashType);
				int v = (Math.abs(hash)) % (M - 1) + 1;
				spq.add(v);
			}
		}

		LOG.debug("M: " + M);
		LOG.debug("smallest hash:" + spq.peek());
		LOG.debug("spq: " + spq.toString());

		if(spq.size() < k) {
			return spq.size();
		}
		else {
			double U_k = (double) spq.poll() / (double) M;
			LOG.debug("U_k : " + U_k);
			double estimate = (double) (k - 1) / U_k;
			LOG.debug("Estimate: " + estimate);
			double ceilEstimate = Math.min(estimate, (double) D);
			LOG.debug("Ceil worst case: " + ceilEstimate);
			return (int) ceilEstimate;
		}
	}

	/**
	 * Deceiving name, but is used to contain the k smallest values inserted.
	 * 
	 * TODO: add utility method to join two partitions
	 * 
	 * TODO: Replace Standard Java Set and Priority Queue with optimized versions.
	 */
	private static class SmallestPriorityQueue {
		private Set<Integer> containedSet;
		private PriorityQueue<Integer> smallestHashes;
		private int k;

		public SmallestPriorityQueue(int k) {
			smallestHashes = new PriorityQueue<>(k, Collections.reverseOrder());
			containedSet = new HashSet<>(1);
			this.k = k;
		}

		public void add(int v) {
			if(!containedSet.contains(v)) {
				if(smallestHashes.size() < k) {
					smallestHashes.add(v);
					containedSet.add(v);
				}
				else if(v < smallestHashes.peek()) {
					LOG.trace(smallestHashes.peek() + " -- " + v);
					smallestHashes.add(v);
					containedSet.add(v);
					containedSet.remove(smallestHashes.poll());
				}
			}
		}

		public int size() {
			return smallestHashes.size();
		}

		public int peek() {
			return smallestHashes.peek();
		}

		public int poll() {
			return smallestHashes.poll();
		}

		@Override
		public String toString() {
			return smallestHashes.toString();
		}
	}
}
