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

public class IGeneratePair<A, B> implements IGenerate<Pair<A, B>> {

	private final IGenerate<A> a;
	private final IGenerate<B> b;

	public IGeneratePair(IGenerate<A> a, IGenerate<B> b) {
		this.a = a;
		this.b = b;
	}

	@Override
	public boolean isEmpty() {
		return a.isEmpty() && b.isEmpty();
	}

	@Override
	public int defaultWaitTime() {
		return Math.max(a.defaultWaitTime(), b.defaultWaitTime());
	}

	@Override
	public Pair<A, B> take() {
		A av = a.take();
		B bv = b.take();
		return new Pair<>(av, bv);
	}

	@Override
	public void generate(int N) throws InterruptedException {
		a.generate(N);
		b.generate(N);
	}

}
