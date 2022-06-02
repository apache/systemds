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

package org.apache.sysds.test.component.federated;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerScalar extends FedWorkerBase {

	private final int rep;
	private final long seed;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker();

		tests.add(new Object[] {port, 1, 12342});
		tests.add(new Object[] {port, 100, 113});

		return tests;
	}

	public FedWorkerScalar(int port, int rep, int seed) {
		super(port);
		this.rep = rep;
		this.seed = seed;
	}

	@Test
	public void verifyPutGetScalar() {
		final Random r = new Random(seed);
		for(int i = 0; i < rep; i++) {
			final double v = r.nextDouble();
			final long id = putDouble(v);
			final double vr = getDouble(id);
			assertEquals("values not equivalent", v, vr, 0.0000001);

		}
	}

	@Test
	public void verifyPutGetSameScalar() {
		final Random r = new Random(seed);
		final long id = putDouble(r.nextDouble());
		final double vrInit = getDouble(id);
		for(int i = 0; i < rep; i++) {
			final double vr = getDouble(id);
			assertEquals("values not equivalent", vrInit, vr, 0.0000001);
		}
	}
}
