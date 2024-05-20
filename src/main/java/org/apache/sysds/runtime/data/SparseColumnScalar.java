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

package org.apache.sysds.runtime.data;

public class SparseColumnScalar extends SparseColumn{


	private static final long serialVersionUID = 2298228659055679831L;

	@Override
	public int size() {
		return 0;
	}

	@Override
	public boolean isEmpty() {
		return false;
	}

	@Override
	public double[] values() {
		return new double[0];
	}

	@Override
	public int[] indexes() {
		return new int[0];
	}

	@Override
	public void reset(int estnns, int maxnns) {

	}

	@Override
	public boolean set(int row, double v) {
		return false;
	}

	@Override
	public boolean add(int row, double v) {
		return false;
	}

	@Override
	public SparseRow append(int row, double v) {
		return null;
	}

	@Override
	public double get(int row) {
		return 0;
	}

	@Override
	public void sort() {

	}

	@Override
	public void compact() {

	}

	@Override
	public void compact(double eps) {

	}

	@Override
	public SparseColumn copy(boolean deep) {
		return null;
	}

	@Override
	public int searchIndexesFirstGTE(int row) {
		return 0;
	}

	@Override
	public int searchIndexesFirstGT(int row) {
		return 0;
	}
}
