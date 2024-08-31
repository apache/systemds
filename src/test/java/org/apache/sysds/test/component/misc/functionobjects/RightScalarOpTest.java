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

import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.junit.Test;

public class RightScalarOpTest {

	@Test
	public void testToString() {
		RightScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 2);
		assertEquals("Right(Plus, 2.0, 1)", op.toString());
	}

	@Test
	public void testToString2() {
		RightScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 2, 2);
		assertEquals("Right(Plus, 2.0, 2)", op.toString());
	}

	@Test
	public void testToString3() {
		RightScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 2341.2, 2);
		assertEquals("Right(Plus, 2341.2, 2)", op.toString());
	}
}
