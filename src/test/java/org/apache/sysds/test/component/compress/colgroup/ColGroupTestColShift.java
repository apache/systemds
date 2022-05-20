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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Basic idea is that we specify a list of compression schemes that a input is allowed to be compressed into. The test
 * verify that these all does the same on a given input. Base on the api for a columnGroup.
 */
@RunWith(value = Parameterized.class)
public class ColGroupTestColShift extends ColGroupBase {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		addConstCases(tests);

		return tests;
	}

	public ColGroupTestColShift(AColGroup base, AColGroup other, int nRow) {
		super(base, other, nRow);
	}

	@Test
	public void shiftColIndices() {
		base.shiftColIndices(3);
		other.shiftColIndices(3);
		assertTrue(Arrays.equals(base.getColIndices(), other.getColIndices()));
	}

}
