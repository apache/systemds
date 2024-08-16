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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.functionobjects.And;
import org.junit.Test;

public class AndTest {

	final And op = And.getAndFnObject();

	@Test
	public void isBool() {
		assertTrue(op.isBinary());
	}

	@Test
	public void isSingleton() {
		assertEquals(op, And.getAndFnObject());
	}

	@Test
	public void e1() {
		assertTrue(op.execute(true, true));
		assertFalse(op.execute(false, true));
		assertFalse(op.execute(true, false));
		assertFalse(op.execute(false, false));
	}

	@Test
	public void t1() {
		assertEquals(0.0, op.execute(0.0, 0.0), 0.0);
	}

	@Test
	public void t2() {
		assertEquals(0.0, op.execute(0.1, 0.0), 0.0);
	}

	@Test
	public void t3() {
		assertEquals(0.0, op.execute(0.0, 0.1), 0.0);
	}

	@Test
	public void t4() {
		assertEquals(1.0, op.execute(0.1, 0.1), 0.0);
	}

	@Test
	public void t5() {
		assertEquals(1.0, op.execute(Double.NaN, 0.1), 0.0);
	}

	@Test
	public void t6() {
		assertEquals(1.0, op.execute(Double.NaN, Double.NaN), 0.0);
	}

	@Test
	public void t7() {
		assertEquals(1.0, op.execute(0.2, Double.NaN), 0.0);
	}

	@Test
	public void t8() {
		assertEquals(1.0, op.execute(0.2, 0.1), 0.0);
	}

	@Test
	public void t9() {
		assertEquals(1.0, op.execute(0.2, 13.1), 0.0);
	}

}
