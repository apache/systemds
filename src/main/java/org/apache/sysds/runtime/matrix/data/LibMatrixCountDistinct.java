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
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator.CountDistinctTypes;
import org.apache.sysds.utils.Hash;
import org.apache.sysds.utils.Hash.HashType;

/**
 * This class contains various methods for counting the number of distinct values inside a MatrixBlock
 */
public class LibMatrixCountDistinct {
	private static final Log LOG = LogFactory.getLog(LibMatrixCountDistinct.class.getName());

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
		if(in.getLength() == 1 || in.isEmpty())
			return 1;
		else if(in.getNonZeros() < minimumSize) {
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
			throw new DMLRuntimeException("Impossible estimate of distinct values");
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
		double[] data;
		long nonZeros = in.getNonZeros();
		if(nonZeros < in.getNumColumns() * in.getNumRows()) {
			distinct.add(0d);
		}
		if(in.sparseBlock == null && in.denseBlock == null) {
			List<ColGroup> colGroups = ((CompressedMatrixBlock) in).getColGroups();
			for(ColGroup cg : colGroups) {
				countDistinctValuesNaive(cg.getValues(), distinct);
			}
		}
		else if(in.sparseBlock != null) {
			SparseBlock sb = in.sparseBlock;

			if(in.sparseBlock.isContiguous()) {
				data = sb.values(0);
				countDistinctValuesNaive(data, distinct);
			}
			else {
				for(int i = 0; i < in.getNumRows(); i++) {
					if(!sb.isEmpty(i)) {
						data = in.sparseBlock.values(i);
						countDistinctValuesNaive(data, distinct);
					}
				}
			}
		}
		else {
			DenseBlock db = in.denseBlock;
			for(int i = 0; i <= db.numBlocks(); i++) {
				data = db.valuesAt(i);
				countDistinctValuesNaive(data, distinct);
			}
		}

		return distinct.size();
	}

	private static Set<Double> countDistinctValuesNaive(double[] valuesPart, Set<Double> distinct) {
		for(double v : valuesPart) {
			distinct.add(v);
		}
		return distinct;
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

		countDistinctValuesKVM(in, op.hashType, k, spq, M);

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
			LOG.debug("Ceil worst case: " + D);
			return (int) ceilEstimate;
		}
	}

	private static void countDistinctValuesKVM(MatrixBlock in, HashType hashType, int k, SmallestPriorityQueue spq,
		int m) {
		double[] data;
		if(in.sparseBlock == null && in.denseBlock == null) {
			List<ColGroup> colGroups = ((CompressedMatrixBlock) in).getColGroups();
			for(ColGroup cg : colGroups) {
				countDistinctValuesKVM(cg.getValues(), hashType, k, spq, m);
			}
		}
		else if(in.sparseBlock != null) {
			SparseBlock sb = in.sparseBlock;
			if(in.sparseBlock.isContiguous()) {
				data = sb.values(0);
				countDistinctValuesKVM(data, hashType, k, spq, m);
			}
			else {
				for(int i = 0; i < in.getNumRows(); i++) {
					if(!sb.isEmpty(i)) {
						data = in.sparseBlock.values(i);
						countDistinctValuesKVM(data, hashType, k, spq, m);
					}
				}
			}
		}
		else {
			DenseBlock db = in.denseBlock;
			final int bil = db.index(0);
			final int biu = db.index(in.rlen);
			for(int i = bil; i <= biu; i++) {
				data = db.valuesAt(i);
				countDistinctValuesKVM(data, hashType, k, spq, m);
			}
		}
	}

	private static void countDistinctValuesKVM(double[] data, HashType hashType, int k, SmallestPriorityQueue spq,
		int m) {
		for(double fullValue : data) {
			int hash = Hash.hash(fullValue, hashType);
			int v = (Math.abs(hash)) % (m - 1) + 1;
			spq.add(v);
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
