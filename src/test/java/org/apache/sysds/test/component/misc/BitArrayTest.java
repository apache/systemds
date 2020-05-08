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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.sysds.runtime.util.BitArray;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class BitArrayTest {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		tests.add(new Object[] {1, 1});
		tests.add(new Object[] {63, 1});
		tests.add(new Object[] {63, 3});
		tests.add(new Object[] {63, 4});
		tests.add(new Object[] {63, 5});
		tests.add(new Object[] {63, 193});
		tests.add(new Object[] {127, 193});
		tests.add(new Object[] {127, 132});
		tests.add(new Object[] {127, 14112});
		tests.add(new Object[] {1000, 14112});
		return tests;
	}

	final long inputLength;
	final Random r;
	final BitArray arr;

	public BitArrayTest(long length, long seed) {
		inputLength = length;
		this.r =  new Random(seed);
		arr = new BitArray(length);
	}

	@Test
	public void testGetterSetter() {
		try{

			// Repeat set and get...
			for(int x = 0; x< inputLength; x++){
				long index = Math.abs(r.nextLong() % inputLength);
				boolean value = arr.get(index);
				arr.set(index, !value);
				assertNotEquals(value, arr.get(index));
			}
		} catch( Exception e){
			e.printStackTrace();
			assertTrue(false);
		}
	}

	public void testLength() {
		// all lengths are padded up to nearest 64 bits.
		long expected = inputLength + 64 - inputLength % 64;
		long actual = arr.getLength();
		assertEquals(expected, actual);
	}

}