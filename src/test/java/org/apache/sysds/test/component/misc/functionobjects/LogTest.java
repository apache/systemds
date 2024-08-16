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

package org.apache.sysds.test.component.misc.functionobjects;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.junit.Test;

public class LogTest {

	protected static final Log LOG = LogFactory.getLog(LogTest.class.getName());
	ValueFunction vf = Builtin.getBuiltinFnObject(BuiltinCode.LOG);

	@Test
	public void t1() {
		double r = vf.execute(2.0, 6.0);
		assertEquals(0.386853d, r, 0.000001);
	}

	@Test
	public void t2() {
		double r = vf.execute(8.0, 6.0);
		assertEquals(1.160558d, r, 0.000001);
	}

	@Test
	public void t2_long() {
		double r = vf.execute(8, 6);
		assertEquals(1.160558d, r, 0.000001);
	}

	@Test
	public void t3() {
		double r = vf.execute(8.0, 0.0);
		assertTrue(Math.abs(r) < 0.00000000001);
	}

	@Test
	public void t3_long() {
		double r = vf.execute(8, 0);
		assertTrue(Math.abs(r) < 0.00000000001);
	}

	@Test
	public void t4() {
		double r = vf.execute(0.0, 1.0);
		assertTrue(Double.isInfinite(r));
	}

	@Test
	public void t4_long() {
		double r = vf.execute(0, 1);
		assertTrue(Double.isInfinite(r));
	}
}
