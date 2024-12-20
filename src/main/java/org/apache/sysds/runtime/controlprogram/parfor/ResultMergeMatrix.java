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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.util.List;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * <p>
 * Due to independence of all iterations, any result has the following properties:
 * </p>
 * 
 * <p>
 * (1) non local var,
 * </p>
 * <p>
 * (2) matrix object, and
 * </p>
 * <p>
 * (3) completely independent.
 * </p>
 * 
 * <p>
 * These properties allow us to realize result merging in parallel without any synchronization.
 * </p>
 */
public abstract class ResultMergeMatrix extends ResultMerge<MatrixObject> {
	private static final long serialVersionUID = 5319002218804570071L;

	public ResultMergeMatrix() {
		super();
	}

	public ResultMergeMatrix(MatrixObject out, MatrixObject[] in, String outputFilename, boolean accum) {
		super(out, in, outputFilename, accum);
	}

	protected void mergeWithoutComp(MatrixBlock out, MatrixBlock in, DenseBlock compare, boolean appendOnly) {
		mergeWithoutComp(out, in, compare, appendOnly, false);
	}

	protected void mergeWithoutComp(MatrixBlock out, MatrixBlock in, DenseBlock compare, boolean appendOnly, boolean par) {
		// pass through to matrix block operations
		if(_isAccum) {
			out.binaryOperationsInPlace(PLUS, in);
			//compare block used for compensation here
			if( compare != null )
				out.binaryOperationsInPlace(MINUS, 
					new MatrixBlock(out.getNumRows(),out.getNumColumns(), compare));
		}
		else {
			MatrixBlock out2 = out.merge(in, appendOnly, par);
			if(out2 != out)
				throw new DMLRuntimeException("Failed merge need to allow returned MatrixBlock to be used");
		}
	}

	/**
	 * NOTE: append only not applicable for with compare because output must be populated with initial state of matrix -
	 * with append, this would result in duplicates.
	 * 
	 * @param out     output matrix block
	 * @param in      input matrix block
	 * @param compare Comparison matrix of old values.
	 */
	protected void mergeWithComp(MatrixBlock out, MatrixBlock in, DenseBlock compare) {
		// Notes for result correctness:
		// * Always iterate over entire block in order to compare all values
		// (using sparse iterator would miss values set to 0)
		// * Explicit NaN awareness because for cases were original matrix contains
		// NaNs, since NaN != NaN, otherwise we would potentially overwrite results
		// * For the case of accumulation, we add out += (new-old) to ensure correct results
		// because all inputs have the old values replicated
		int rows = in.getNumRows();
		int cols = in.getNumColumns();
		if(in.isEmptyBlock(false))
			mergeWithCompEmpty(out, rows, cols, compare);
		else if(in.isInSparseFormat())
			mergeSparse(out, in, rows, cols, compare);
		else // DENSE
			mergeGeneric(out, in, rows, cols, compare);
	}

	private void mergeWithCompEmpty(MatrixBlock out, int m, int n, DenseBlock compare) {
		for(int i = 0; i < m; i++)
			mergeWithCompEmptyRow(out, m, n, compare, i);
	}

	private void mergeWithCompEmptyRow(MatrixBlock out, int m, int n, DenseBlock compare, int i) {
		for(int j = 0; j < n; j++) {
			final double valOld = compare.get(i, j);
			if(!equals(0.0, valOld)) // NaN awareness
				out.set(i, j, 0);
		}
	}

	private void mergeSparse(MatrixBlock out, MatrixBlock in, int m, int n, DenseBlock compare) {
		final SparseBlock a = in.getSparseBlock();
		for(int i = 0; i < m; i++) {
			if(a.isEmpty(i))
				mergeWithCompEmptyRow(out, m, n, compare, i);
			else {
				int apos = a.pos(i);
				int alen = a.size(i) + apos;
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				int j = 0;
				for(; j < n && apos < alen; j++) { // inside
					boolean aposValid = (aix[apos] == j);
					double valOld = compare.get(i, j);
					double valNew = aposValid ? avals[apos] : 0.0;
					if(!equals(valNew, valOld)) { // NaN awareness
						double value = !_isAccum ? valNew : (out.get(i, j) + (valNew - valOld));
						out.set(i, j, value);
					}
					if(aposValid)
						apos++;
				}
				for(; j < n; j++) {
					double valOld = compare.get(i, j);
					if(valOld != 0) {
						double value = (out.get(i, j) - valOld);
						out.set(i, j, value);
					}
				}
			}
		}
	}

	private void mergeGeneric(MatrixBlock out, MatrixBlock in, int m, int n, DenseBlock compare) {
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				final double valOld = compare.get(i, j);
				final double valNew = in.get(i, j); // input value
				if(!equals(valNew, valOld)) { // NaN awareness
					out.set(i, j, valNew);
				}
			}
		}
	}
	
	private boolean equals(double valNew, double valOld) {
		return (valNew == valOld && !Double.isNaN(valNew))     //for changed values 
			|| (Double.isNaN(valNew) && Double.isNaN(valOld)); //NaN awareness 
	}

	protected long computeNonZeros(MatrixObject out, List<MatrixObject> in) {
		// sum of nnz of input (worker result) - output var existing nnz
		long outNNZ = out.getDataCharacteristics().getNonZeros();
		return outNNZ - in.size() * outNNZ + in.stream()//
			.mapToLong(m -> m.getDataCharacteristics().getNonZeros()).sum();
	}
}
