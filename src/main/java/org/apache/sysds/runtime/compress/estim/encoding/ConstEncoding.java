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

package org.apache.sysds.runtime.compress.estim.encoding;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;

/** Const encoding for cases where the entire group of columns is the same value */
public class ConstEncoding extends AEncode {

	private final int[] counts;

	protected ConstEncoding(int nRows) {
		this.counts = new int[] {nRows};
	}

	@Override
	public IEncode combine(IEncode e) {
		return e;
	}

	@Override
	public Pair<IEncode, HashMapLongInt> combineWithMap(IEncode e) {
		if(e instanceof EmptyEncoding)
			return new ImmutablePair<>(this, null);
		else
			return new ImmutablePair<>(e, null);
	}

	@Override
	public int getUnique() {
		return 1;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		return sb.toString();
	}

	@Override
	public EstimationFactors extractFacts(int nRows, double tupleSparsity, double matrixSparsity,
		CompressionSettings cs) {
		return new EstimationFactors(1, nRows, nRows, counts, 0, nRows, 1, false, false, matrixSparsity, tupleSparsity);
	}

	@Override
	public boolean isDense() {
		return true;
	}

	@Override
	public boolean equals(IEncode e) {
		return e instanceof ConstEncoding && ((ConstEncoding) e).counts.length == this.counts.length;
	}
}
