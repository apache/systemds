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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.utils.MemoryEstimates;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.openjdk.jol.datamodel.X86_64_DataModel;
import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.layouters.HotSpotLayouter;
import org.openjdk.jol.layouters.Layouter;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class MemoryEstimateTest {

	private enum ArrayType {
		BYTE, // RLE
		CHAR, // OLE
		INT, // UC
		DOUBLE, // RLE
	}

	@Parameterized.Parameters
	public static Iterable<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(ArrayType x : ArrayType.values()) {
			tests.add(new Object[] {x, 1});
		}
		Random r = new Random(123812415123L);
		for(ArrayType x : ArrayType.values()) {
			tests.add(new Object[] {x, r.nextInt(1000)});
			tests.add(new Object[] {x, r.nextInt(10000)});
			tests.add(new Object[] {x, r.nextInt(100000)});
		}
		return tests;
	}

	@Parameterized.Parameter
	public ArrayType arrayToMeasure;
	@Parameterized.Parameter(1)
	public int length;

	public Layouter l = new HotSpotLayouter(new X86_64_DataModel());

	@Test
	public void test() {
		switch(arrayToMeasure) {
			case BYTE:
				byte[] arrayByte = new byte[length];
				assertEquals(MemoryEstimates.byteArrayCost(length), measure(arrayByte), 0.2);
				break;
			case CHAR:
				char[] arrayChar = new char[length];
				assertEquals(MemoryEstimates.charArrayCost(length), measure(arrayChar), 0.2);
				break;
			case INT:
				int[] arrayInt = new int[length];
				assertEquals(MemoryEstimates.intArrayCost(length), measure(arrayInt), 0.2);
				break;
			case DOUBLE:
				double[] arrayDouble = new double[length];
				assertEquals(MemoryEstimates.doubleArrayCost(length), measure(arrayDouble), 0.2);
				break;
			default:
				throw new NotImplementedException(arrayToMeasure + " not implemented");
		}
	}

	private long measure(Object obj) {
		return ClassLayout.parseInstance(obj, l).instanceSize() + 8;
	}
}
