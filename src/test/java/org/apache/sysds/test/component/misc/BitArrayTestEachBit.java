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

import org.apache.sysds.runtime.util.BitArray;
import org.junit.Assert;
import org.junit.Test;

public class BitArrayTestEachBit {

	@Test
	public void test() {
		BitArray arr = new BitArray(63);
		long expected = 0;
		long actual = arr.getChunk(0);

		Assert.assertEquals(expected,actual);
		expected = 1;
		for(int x = 0; x < 64; x++){
			arr.set(x, true);
			actual = arr.getChunk(0);
			Assert.assertEquals(expected,actual);
			expected = expected << 1L;
			arr.set(x, false);
		}
	}

	@Test
	public void test2() {
		BitArray arr = new BitArray(127);
		long expected_C1 = 0;
		long actual_C1 = arr.getChunk(0);
		long expected_C2 = 0;
		long actual_C2 = arr.getChunk(1);

		Assert.assertEquals(expected_C1,actual_C1);
		Assert.assertEquals(expected_C2,actual_C2);
		expected_C1 = 1;
		for(int x = 0; x < 128; x++){
			arr.set(x, true);
			if(x == 64){
				expected_C1 = 0L;
				expected_C2 = 1;
			}
			actual_C1 = arr.getChunk(0);
			actual_C2 = arr.getChunk(1);
			System.out.println(arr.toString());
			Assert.assertEquals(expected_C1,actual_C1);
			Assert.assertEquals(expected_C2,actual_C2);

			if(x< 64){
				expected_C1 = expected_C1 << 1L;
			}else{
				expected_C2 = expected_C2 << 1L;
			}
			
			arr.set(x, false);
		}
	}

}