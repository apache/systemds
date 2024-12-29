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

package org.apache.sysds.test.component.compress.combine;

import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;
import org.junit.Test;

public class CombineEncodings {

	public static final Log LOG = LogFactory.getLog(CombineEncodings.class.getName());

	@Test
	public void combineCustom() {
		IEncode ae = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10));
		IEncode be = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10));
		Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
		IEncode ce = cec.getLeft();
		HashMapLongInt cem = cec.getRight();
		assertTrue(cem.size() == 10);
		assertTrue(cem.size() == ce.getUnique());
		assertTrue(ce.equals(new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10))));

	}

	@Test
	public void combineCustom2() {
		IEncode ae = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 8}, 10));
		IEncode be = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10));
		Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
		IEncode ce = cec.getLeft();
		HashMapLongInt cem = cec.getRight();
		assertTrue(cem.size() == 10);
		assertTrue(cem.size() == ce.getUnique());
		assertTrue(ce.equals(new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10))));

	}

	@Test
	public void combineCustom3() {
		IEncode ae = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 8}, 10));
		IEncode be = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 9}, 10));
		Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
		IEncode ce = cec.getLeft();
		HashMapLongInt cem = cec.getRight();
		assertTrue(cem.size() == 9);
		assertTrue(cem.size() == ce.getUnique());
		assertTrue(ce.equals(new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 8}, 9))));

	}

	@Test
	public void combineCustom4() {
		// same mapping require the unique to be correct!!
		IEncode ae = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 0}, 8));
		IEncode be = new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 0}, 8));
		Pair<IEncode, HashMapLongInt> cec = ae.combineWithMap(be);
		IEncode ce = cec.getLeft();
		HashMapLongInt cem = cec.getRight();
		assertTrue(cem.size() == 8);
		assertTrue(cem.size() == ce.getUnique());
		assertTrue(ce.equals(new DenseEncoding(MapToFactory.create(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 7, 0}, 8))));
	}
}
