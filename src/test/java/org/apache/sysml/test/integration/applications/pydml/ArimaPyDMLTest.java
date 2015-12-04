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

package org.apache.sysml.test.integration.applications.pydml;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import org.apache.sysml.test.integration.applications.ArimaTest;

@RunWith(value = Parameterized.class)
public class ArimaPyDMLTest extends ArimaTest {

	public ArimaPyDMLTest(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi) {
		super(m, p, d, q, P, D, Q, s, include_mean, useJacobi);
	}

	@Test
	public void testArimaPyDml() {
		testArima(ScriptType.PYDML);
	}

}
