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

package org.apache.sysds.test.component.compress.mapping;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.Random;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToBit;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory.MAP_TYPE;
import org.junit.Test;

public class CustomMappingTest {

	int[] data = new int[] {0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	@Test
	public void createBinary() {
		try {

			CompressedMatrixBlock.debug = true;
			MapToFactory.create(data, 2);
			MapToFactory.create(127, data, 2);
		}
		catch(RuntimeException e) {
			e.printStackTrace();
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void equalsTest1() {
		int[] in = new int[] {1, 2, 3, 4};
		AMapToData d = MapToFactory.create(in, 5);
		AMapToData e = MapToFactory.create(in, 5);

		assertTrue(d.equals(e));

	}

	@Test
	public void equalsTest2() {
		AMapToData d = MappingTestUtil.createRandomMap(100, 2, new Random(23));
		AMapToData e = MappingTestUtil.createRandomMap(100, 2, new Random(23));

		assertTrue(d.equals(e));

	}

	@Test
	public void equalsTest3() {
		AMapToData d = MappingTestUtil.createRandomMap(20, 2, new Random(23));
		AMapToData e = MappingTestUtil.createRandomMap(20, 2, new Random(23));

		assertTrue(d.equals(e));

	}

	@Test
	public void equalsTest4() {
		AMapToData d = MappingTestUtil.createRandomMap(20, 2, new Random(23));
		AMapToData e = MappingTestUtil.createRandomMap(20, 2, new Random(22));

		assertFalse(d.equals(e));

	}

	@Test
	public void equalsTest5() {
		AMapToData d = MappingTestUtil.createRandomMap(102, 2, new Random(23));
		AMapToData e = MappingTestUtil.createRandomMap(102, 2, new Random(22));

		assertFalse(d.equals(e));

	}

	@Test
	public void empty() {
		MapToBit d = (MapToBit) MapToFactory.create(100, MAP_TYPE.BIT);

		assertTrue(d.isEmpty());

		d.set(23, 1);
		assertFalse(d.isEmpty());

	}

	@Test
	public void verifyInvalid() {

		AMapToData d = MapToFactory.create(2, 2);
		CompressedMatrixBlock.debug = true;
		AMapToData spy = spy(d);
		when(spy.getIndex(anyInt())).thenReturn(32);
		Exception e = assertThrows(DMLCompressionException.class, () -> spy.verify());

		assertTrue(e.getMessage().equals("Invalid construction of Mapping data containing values above unique"));
	}

	@Test
	public void equalsObject() {

		AMapToData d = MapToFactory.create(2, 2);
		assertFalse(d.equals(new Object()));
	}

	@Test
	public void equalsObjectTrue() {

		AMapToData d = MapToFactory.create(2, 2);
		assertTrue(d.equals((Object) d));
	}

	@Test
	public void equalsObjectOtherTrue() {

		AMapToData d = MapToFactory.create(2, 2);
		AMapToData e = MapToFactory.create(2, 2);
		assertTrue(d.equals((Object) e));
	}

	@Test
	public void equalsObjectOtherFalse() {
		AMapToData d = MapToFactory.create(2, 2);
		AMapToData e = MapToFactory.create(2, 2);
		e.set(0, 1);
		assertFalse(d.equals((Object) e));
	}

	@Test
	public void numberRuns1() {

		AMapToData d = MapToFactory.create(2, 2);
		assertEquals(1, d.countRuns());
	}

	@Test
	public void numberRuns2() {

		AMapToData d = MapToFactory.create(new int[] {1, 1, 1, 2, 2, 2}, 2);
		assertEquals(2, d.countRuns());
	}

	@Test
	public void numberRuns3() {

		AMapToData d = MapToFactory.create(new int[] {1, 1, 1, 2, 1, 2}, 2);
		assertEquals(4, d.countRuns());
	}
}
