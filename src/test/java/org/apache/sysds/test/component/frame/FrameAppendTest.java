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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FrameAppendTest {
	public FrameBlock f;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			for(int i = 0; i < 10; i++) {
				tests.add(new Object[] {TestUtils.generateRandomFrameBlock(300, 300, i)});
				tests.add(new Object[] {TestUtils.generateRandomFrameBlock(100, 10, i)});
				tests.add(new Object[] {TestUtils.generateRandomFrameBlock(10, 10, i)});
				tests.add(new Object[] {TestUtils.generateRandomFrameBlock(1, 1, i)});
				tests.add(new Object[] {TestUtils.generateRandomFrameBlock(1, 10, i)});
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public FrameAppendTest(FrameBlock f) {
		this.f = f;
	}

	@Test
	public void appendSelfRBind() {
		FrameBlock ff = f.append(f, false);
		final int nRow = f.getNumRows();
		for(int r = 0; r < ff.getNumRows(); r++)
			for(int c = 0; c < ff.getNumColumns(); c++)
				assertEquals(ff.get(r, c).toString(), f.get(r % nRow, c).toString());
	}

	@Test
	public void appendSelfCBind() {
		FrameBlock ff = f.append(f, true);
		final int nCol = f.getNumColumns();
		for(int r = 0; r < ff.getNumRows(); r++)
			for(int c = 0; c < ff.getNumColumns(); c++)
				assertEquals(ff.get(r, c).toString(), f.get(r, c % nCol).toString());
	}

	// append other type

	// append null block.

}
