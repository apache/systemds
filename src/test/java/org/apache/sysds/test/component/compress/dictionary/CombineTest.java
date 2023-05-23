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

import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.junit.Test;

public class CombineTest {
	@Test
	public void singleBothSides() {
		try {

			ADictionary a = Dictionary.create(new double[] {1.2});
			ADictionary b = Dictionary.create(new double[] {1.4});

			ADictionary c = DictionaryFactory.combineDense(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void singleOneSideBothSides() {
		try {
			ADictionary a = Dictionary.create(new double[] {1.2, 1.3});
			ADictionary b = Dictionary.create(new double[] {1.4});

			ADictionary c = DictionaryFactory.combineDense(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void twoBothSides() {
		try {
			ADictionary a = Dictionary.create(new double[] {1.2, 1.3});
			ADictionary b = Dictionary.create(new double[] {1.4, 1.5});

			ADictionary c = DictionaryFactory.combineDense(a, 1, b, 1);

			assertEquals(c.getValue(0, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(0, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(1, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(1, 1, 2), 1.4, 0.0);
			assertEquals(c.getValue(2, 0, 2), 1.2, 0.0);
			assertEquals(c.getValue(2, 1, 2), 1.5, 0.0);
			assertEquals(c.getValue(3, 0, 2), 1.3, 0.0);
			assertEquals(c.getValue(3, 1, 2), 1.5, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
