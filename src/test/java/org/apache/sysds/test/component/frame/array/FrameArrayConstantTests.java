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

package org.apache.sysds.test.component.frame.array;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FrameArrayConstantTests {
	protected static final Log LOG = LogFactory.getLog(FrameArrayConstantTests.class.getName());

	final public ValueType t;
	final public int nRow;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		try {
			for(ValueType t : ValueType.values()) {
				if(t == ValueType.UNKNOWN)
					continue;
				tests.add(new Object[] {t, 10});
				tests.add(new Object[] {t, 100});
				tests.add(new Object[] {t, 1});
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public FrameArrayConstantTests(ValueType t, int nRow) {
		this.t = t;
		this.nRow = nRow;
	}

	@Test
	public void testConstruction() {
		try {
			Array<?> a = ArrayFactory.allocate(t, nRow, "0");
			if(a.getValueType() == ValueType.CHARACTER)

				for(int i = 0; i < nRow; i++)
					assertEquals(a.getAsDouble(i), 48.0, 0.0000000001);
			else
				for(int i = 0; i < nRow; i++)
					assertEquals(a.getAsDouble(i), 0.0, 0.0000000001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testConstruction_default() {
		try {
			Array<?> a = ArrayFactory.allocate(t, nRow);
			if(t != ValueType.STRING && t != ValueType.CHARACTER)
				for(int i = 0; i < nRow; i++)
					assertEquals(a.getAsDouble(i), 0.0, 0.0000000001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testConstruction_1() {
		try {
			if(t == ValueType.HASH64)
				return;
			Array<?> a = ArrayFactory.allocate(t, nRow, "1.0");
			for(int i = 0; i < nRow; i++)
				assertEquals(a.getAsDouble(i), 1.0, 0.0000000001);
		}
		catch(NumberFormatException e){
			// this is okay to throw
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testConstruction_null() {
		try {
			Array<?> a = ArrayFactory.allocate(t, nRow, null);
			if(t != ValueType.STRING && t != ValueType.CHARACTER)
				for(int i = 0; i < nRow; i++)
					assertEquals(a.getAsDouble(i), 0.0, 0.0000000001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
