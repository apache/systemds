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

package org.apache.sysds.test.component.compress.dictionary;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import scala.util.Random;

@RunWith(value = Parameterized.class)
public class DictionaryTests {

	private final int nRow;
	private final int nCol;
	private final ADictionary a;
	private final ADictionary b;

	public DictionaryTests(ADictionary a, ADictionary b, int nRow, int nCol) {
		this.nRow = nRow;
		this.nCol = nCol;
		this.a = a;
		this.b = b;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			addAll(tests, new double[] {1, 2, 3, 4, 5}, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static void addAll(List<Object[]> tests, double[] vals, int cols) {
		tests.add(new Object[] {Dictionary.create(vals), MatrixBlockDictionary.createDictionary(vals, cols, true),
			vals.length / cols, cols});
	}

	@Test
	public void sum() {
		int[] counts = new int[nCol];
		Random r = new Random(134);
		for(int i = 0; i < nCol; i++) {
			counts[i] = r.nextInt(100);
		}

		double as = a.sum(counts, nCol);
		double bs = b.sum(counts, nCol);
		assertEquals(as, bs, 0.0000001);
	}
}
