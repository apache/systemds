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

package org.apache.sysds.test.component.compress.offset;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetByte;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetChar;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class OffsetTests {

	private static final long sizeTolerance = 265;

	public int[] data;
	public OFF_TYPE type;
	private AOffset o;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		// It is assumed that the input is in sorted order, all values are positive and there are no duplicates.
		for(OFF_TYPE t : OFF_TYPE.values()) {
			tests.add(new Object[] {new int[] {1, 2}, t});
			tests.add(new Object[] {new int[] {2, 142}, t});
			tests.add(new Object[] {new int[] {142, 421}, t});
			tests.add(new Object[] {new int[] {1, 1023}, t});
			tests.add(new Object[] {new int[] {1023, 1024}, t});
			tests.add(new Object[] {new int[] {1023}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {0}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {Character.MAX_VALUE, ((int) Character.MAX_VALUE) * 2}, t});
			tests.add(new Object[] {new int[] {0, 256}, t});
			tests.add(new Object[] {new int[] {0, 254}, t});
			tests.add(new Object[] {new int[] {0, Character.MAX_VALUE}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) + 1}, t});
			tests.add(new Object[] {new int[] {0, ((int) Character.MAX_VALUE) - 1}, t});
			tests.add(new Object[] {new int[] {0, 256 * 2}, t});
			tests.add(new Object[] {new int[] {0, 255 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 2}, t});
			tests.add(new Object[] {new int[] {0, 254 * 3}, t});
			tests.add(new Object[] {new int[] {0, 255 * 3}, t});
			tests.add(new Object[] {new int[] {0, 256 * 3}, t});
			tests.add(new Object[] {new int[] {255 * 3, 255 * 5}, t});
			tests.add(new Object[] {new int[] {1000000, 1000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {100000000, 100000000 + 255 * 5}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 255 * 4, 1500}, t});
			tests.add(new Object[] {new int[] {0, 1, 2, 3, 4, 5}, t});
			tests.add(new Object[] {new int[] {2458248, 2458249, 2458253, 2458254, 2458256, 2458257, 2458258, 2458262,
				2458264, 2458266, 2458267, 2458271, 2458272, 2458275, 2458276, 2458281}, t});
		}
		return tests;
	}

	public OffsetTests(int[] data, OFF_TYPE type) {
		this.data = data;
		this.type = type;
		switch(type) {
			case BYTE:
				this.o = new OffsetByte(data);
				break;
			case CHAR:
				this.o = new OffsetChar(data);
				break;
			default:
				throw new NotImplementedException("not implemented");
		}
	}

	@Test
	public void testConstruction() {
		try {
			AIterator i = o.getIterator();
			compare(i, data);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testSerialization() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			o.write(fos);

			// Serialize in
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);

			AOffset n = OffsetFactory.readIn(fis);

			AIterator i = n.getIterator();
			compare(i, data);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testGetSize() {
		assertEquals(data.length, o.getSize());
	}

	@Test
	public void testOnDiskSizeInBytes() {
		try {
			// Serialize out
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			o.write(fos);

			int size = bos.toByteArray().length;
			assertEquals(size, o.getExactSizeOnDisk());
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testInMemoryEstimateIsSameAsActualOrSmaller() {
		try {
			long inMemorySize = o.getInMemorySize();
			long estimatedSize;
			switch(type) {
				case BYTE:
					estimatedSize = OffsetByte.getInMemorySize(data.length);
					break;
				case CHAR:
					estimatedSize = OffsetChar.getInMemorySize(data.length);
					break;
				default:
					throw new DMLCompressionException("Unknown input");
			}
			final String errorMessage = "in memory size: " + inMemorySize + " is not smaller than estimate: "
				+ estimatedSize + " with tolerance " + sizeTolerance;
			assertTrue(errorMessage, inMemorySize - sizeTolerance <= estimatedSize);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	private void compare(AIterator i, int[] v) {
		for(int j = 0; j < v.length; j++) {

			if(v[j] != i.value())
				fail("incorrect result using : " + o.getClass().getSimpleName() + " expected: " + Arrays.toString(v)
					+ " but was :" + o.toString());
			if(i.hasNext())
				i.next();
		}
	}

}
