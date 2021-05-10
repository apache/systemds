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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * A sparse dictionary implementation, use if the tuples are sparse.
 */
public class SparseDictionary extends ADictionary {

	@Override
	public double[] getValues() {
		LOG.warn("Inefficient materialization of sparse Dictionary.");

		return null;
	}

	@Override
	public double getValue(int i) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int hasZeroTuple(int nCol) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getInMemorySize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double aggregate(double init, Builtin fn) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] aggregateTuples(Builtin fn, int nCol) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public ADictionary apply(ScalarOperator op) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary applyBinaryRowOpLeft(ValueFunction fn, double[] v, boolean sparseSafe, int[] colIndexes) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary applyBinaryRowOpRight(ValueFunction fn, double[] v, boolean sparseSafe, int[] colIndexes) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary clone() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary cloneAndExtend(int len) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isLossy() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int getNumberOfValues(int ncol) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] sumAllRowsToDouble(boolean square, int nrColumns) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double sumRow(int k, boolean square, int nrColumns) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] colSum(int[] counts, int nCol) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void colSum(double[] c, int[] counts, int[] colIndexes, boolean square) {
		// TODO Auto-generated method stub

	}

	@Override
	public double sum(int[] counts, int ncol) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double sumsq(int[] counts, int ncol) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String getString(int colIndexes) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void addMaxAndMin(double[] ret, int[] colIndexes) {
		// TODO Auto-generated method stub

	}

	@Override
	public ADictionary sliceOutColumnRange(int idxStart, int idxEnd, int previousNumberOfColumns) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary reExpandColumns(int max) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean containsValue(double pattern) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long getNumberNonZerosContained() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void addToEntry(Dictionary d, int fr, int to, int nCol) {
		// TODO Auto-generated method stub

	}

	@Override
	public double[] getMostCommonTuple(int[] counts, int nCol) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public ADictionary subtractTuple(double[] tuple) {
		// TODO Auto-generated method stub
		return null;
	}

}
