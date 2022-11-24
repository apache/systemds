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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class ColumnMetadataTests {

	public ColumnMetadata d;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			tests.add(new Object[] {new ColumnMetadata()});
			tests.add(new Object[] {new ColumnMetadata(3241)});
			tests.add(new Object[] {new ColumnMetadata(32)});
			tests.add(new Object[] {new ColumnMetadata(32L, "")});
			tests.add(new Object[] {new ColumnMetadata(-1, "")});
			tests.add(new Object[] {new ColumnMetadata(-1, "131")});
			tests.add(new Object[] {new ColumnMetadata(32, "Hi")});
			tests.add(new Object[] {new ColumnMetadata(32L, "something")});
			tests.add(new Object[] {new ColumnMetadata(32, null)});
			tests.add(new Object[] {new ColumnMetadata(new ColumnMetadata(32, null))});
			tests.add(new Object[] {new ColumnMetadata(new ColumnMetadata())});
			tests.add(new Object[] {new ColumnMetadata(new ColumnMetadata(333))});
			tests.add(new Object[] {new ColumnMetadata(new ColumnMetadata(3, "others"))});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public ColumnMetadataTests(ColumnMetadata d) {
		this.d = d;

	}

	@Test
	public void copyConstructor() {
		assertTrue(d.equals(new ColumnMetadata(d)));
	}

	@Test
	public void notEquals_1() {
		ColumnMetadata b = new ColumnMetadata(d);
		b.setNumDistinct(b.getNumDistinct() + 132415L);
		assertFalse(d.equals(b));
	}

	@Test
	public void notEquals_2() {
		if(d.getMvValue() != null && !d.getMvValue().equals("")) {
			ColumnMetadata b = new ColumnMetadata(d);
			b.setMvValue("");
			assertFalse(d.equals(b));
		}
	}

	@Test
	public void notEquals_3() {
		try {
			if(d.getMvValue() == null || !d.getMvValue().equals("a")) {
				ColumnMetadata b = new ColumnMetadata(d);
				b.setMvValue("a");
				assertFalse(d.equals(b));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void equalsObject() {
		assertTrue(d.equals((Object) new ColumnMetadata(d)));
		assertFalse(d.equals(1324));
	}

	@Test
	public void getSetDistinct() {
		ColumnMetadata dd = new ColumnMetadata(d);
		long before = dd.getNumDistinct();
		long set = 12355555L + before;
		dd.setNumDistinct(12355555L + before);
		long after = dd.getNumDistinct();
		assertTrue(after == set);
	}

	@Test
	public void getSetDistinctNegative() {
		ColumnMetadata dd = new ColumnMetadata(d);
		dd.setNumDistinct(-1324142L);
		long after = dd.getNumDistinct();
		assertTrue(after == -1);
	}

	@Test
	public void getSetMyValue() {
		ColumnMetadata dd = new ColumnMetadata(d);
		String before = dd.getMvValue();
		String set = 12355555L + before;
		dd.setMvValue(12355555L + before);
		String after = dd.getMvValue();
		assertTrue(after.equals(set));
	}

	@Test
	public void getSetEmptyMyValue() {
		ColumnMetadata dd = new ColumnMetadata(d);
		String before = dd.getMvValue();
		String set = 12355555L + before;
		dd.setMvValue(12355555L + before);
		String after = dd.getMvValue();
		assertTrue(after.equals(set));
	}

	@Test
	public void toStringTest() {
		// just verify we can get a string
		d.toString();
	}

	@Test
	public void testGetInMemorySize() {
		ColumnMetadata dd = new ColumnMetadata(d);
		long m = dd.getInMemorySize();
		dd.setMvValue(d.getMvValue() + "HHH");
		long m2 = dd.getInMemorySize();
		assertTrue(16 < m);
		assertTrue(m < m2);
	}

	@Test
	public void isDefault() {
		assertTrue(new ColumnMetadata().isDefault());
		if(d.getNumDistinct() > 0)
			assertFalse(d.isDefault());
	}

	@Test
	public void serialize() {
		assertTrue(d.equals(serializeAndBack(d)));
	}

	public static ColumnMetadata serializeAndBack(ColumnMetadata a) {
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			a.write(fos);
			DataInputStream fis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
			return ColumnMetadata.read(fis);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Error in io", e);
		}
	}
}
