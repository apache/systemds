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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.junit.Test;

public class FrameCreationTest {
	@Test
	public void createEmptyRow() {
		FrameBlock a = createEmptyRow(new ValueType[] {ValueType.STRING, ValueType.STRING});
		assertEquals(null, a.get(0, 0));
		assertEquals(null, a.get(0, 1));
	}

	private FrameBlock createEmptyRow(ValueType[] schema) {
		String[][] arr = new String[1][];
		arr[0] = new String[schema.length];
		return new FrameBlock(schema, null, arr);
	}
}
