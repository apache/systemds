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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.junit.Test;

public class NegativeArrayTests {

	@Test(expected = DMLRuntimeException.class)
	public void testAllocateInvalidArray() {
		ArrayFactory.allocate(ValueType.UNKNOWN, 1324);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testEstimateMemorySizeInvalid() {
		ArrayFactory.getInMemorySize(ValueType.UNKNOWN, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testChangeTypeToInvalid() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3});
		a.changeType(ValueType.UNKNOWN);
	}

	@Test(expected = NotImplementedException.class)
	public void testChangeTypeToUInt8() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3});
		a.changeType(ValueType.UINT8);
	}

	@Test(expected = DMLRuntimeException.class)
	public void getMinMax() {
		ArrayFactory.create(new int[] {1, 2, 3, 4}).getMinMaxLength();
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_1() {
		StringArray a = ArrayFactory.create(new String[] {"1", "10", "0"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_2() {
		StringArray a = ArrayFactory.create(new String[] {"1", "-1", "0"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_3() {
		StringArray a = ArrayFactory.create(new String[] {"HI", "false", "0"});
		a.changeType(ValueType.BOOLEAN);
	}
}
