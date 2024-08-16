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

package org.apache.sysds.performance.generators;

import org.apache.sysds.runtime.matrix.data.Pair;

public class GenPair<T> implements IGenerate<Pair<T, T>> {

	private IGenerate<T> g1;
	private IGenerate<T> g2;

	public GenPair(IGenerate<T> g1, IGenerate<T> g2) {
		this.g1 = g1;
		this.g2 = g2;
	}

	@Override
	public boolean isEmpty() {
		return g1.isEmpty() || g2.isEmpty();
	}

	@Override
	public int defaultWaitTime() {
		return g1.defaultWaitTime() + g2.defaultWaitTime();
	}

	@Override
	public Pair<T, T> take() {
		return new Pair<>(g1.take(), g2.take());
	}

	@Override
	public void generate(int N) throws InterruptedException {
		g1.generate(N);
		g2.generate(N);
	}

}
