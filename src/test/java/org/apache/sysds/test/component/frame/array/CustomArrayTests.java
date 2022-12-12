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

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.LongArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.junit.Test;

public class CustomArrayTests {

	@Test
	public void getMinMax_1() {
		StringArray a = ArrayFactory.create(new String[] {"a", "aa", "aaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 1);
		assertTrue(mm.getValue() == 3);
	}

	@Test
	public void getMinMax_2() {
		StringArray a = ArrayFactory.create(new String[] {"", null, "aaaaaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 0);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void getMinMax_3() {
		StringArray a = ArrayFactory.create(new String[] {null, null, "aaaaaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 6);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void getMinMax_4() {
		StringArray a = ArrayFactory.create(new String[] {"aaaaaa", null, "null"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 4);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void changeType() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "0"});
		IntegerArray ai = (IntegerArray) a.changeType(ValueType.INT32);
		assertTrue(ai.get(0) == 1);
		assertTrue(ai.get(1) == 2);
		assertTrue(ai.get(2) == 0);
	}

	@Test
	public void changeTypeLong() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "0"});
		LongArray ai = (LongArray) a.changeType(ValueType.INT64);
		assertTrue(ai.get(0) == 1);
		assertTrue(ai.get(1) == 2);
		assertTrue(ai.get(2) == 0);
	}

	@Test
	public void changeTypeBoolean() {
		StringArray a = ArrayFactory.create(new String[] {"1", "0", "0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void analyzeValueTypeStringBoolean(){
		StringArray a = ArrayFactory.create(new String[]{"1", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero(){
		StringArray a = ArrayFactory.create(new String[]{"1.0", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_2(){
		StringArray a = ArrayFactory.create(new String[]{"1.00", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_3(){
		StringArray a = ArrayFactory.create(new String[]{"1.00000000000", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringInt32(){
		StringArray a = ArrayFactory.create(new String[]{"13", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}


	@Test
	public void analyzeValueTypeStringInt32_withPointZero(){
		StringArray a = ArrayFactory.create(new String[]{"13.0", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero_2(){
		StringArray a = ArrayFactory.create(new String[]{"13.0000", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}


	@Test
	public void analyzeValueTypeStringInt32_withPointZero_3(){
		StringArray a = ArrayFactory.create(new String[]{"13.00000000000000", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}


	@Test
	public void analyzeValueTypeStringInt64(){
		StringArray a = ArrayFactory.create(new String[]{""+ (((long)Integer.MAX_VALUE)  + 10L), "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT64);
	}


	@Test
	public void analyzeValueTypeStringFP32(){
		StringArray a = ArrayFactory.create(new String[]{"132", "131.1", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.FP32);
	}

	@Test
	public void analyzeValueTypeStringFP64(){
		StringArray a = ArrayFactory.create(new String[]{"132", "131.0012345678912345", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.FP64);
	}

	@Test
	public void analyzeValueTypeStringFP32_string(){
		StringArray a = ArrayFactory.create(new String[]{"\"132\"", "131.1", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.FP32);
	}

	@Test
	public void setRangeBitSet(){
		
	}
}
