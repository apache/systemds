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

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

public class FrameToStringTest {
	@Test
	public void testDefault() {
		FrameBlock f = createFrameBlock();
		assertTrue(DataConverter.toString(f).length() < 75);
	}
	
	@Test
	public void test100x100() {
		FrameBlock f = createFrameBlock();
		assertTrue(DataConverter.toString(f, false, " ", "\n", 100, 100, 3).length() < 75);
	}
	
	private FrameBlock createFrameBlock() {
		FrameBlock f = new FrameBlock(new ValueType[]{ValueType.STRING, ValueType.STRING});
		for(int i=0; i<5; i++)
			f.appendRow(new String[] {"a","b"});
		return f;
	}
}
