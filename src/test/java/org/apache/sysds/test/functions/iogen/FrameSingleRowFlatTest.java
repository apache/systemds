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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.common.Types;
import org.junit.Test;

public class FrameSingleRowFlatTest extends GenerateReaderFrameTest {

	private final static String TEST_NAME = "FrameSingleRowFlatTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}


	// CSV: Frame
	// 1. dataset contain INT32 values
	@Test
	public void test1() {
		sampleRaw = "1,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32};
		runGenerateReaderTest(false);
	}

	// 2. dataset contain different value types
	@Test
	public void test2() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32};
		runGenerateReaderTest(false);
	}

	@Test
	public void test3() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest(false);
	}

	@Test
	public void test4() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2", "b"}, {"6", "7", "bb"}, {"11", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest(false);
	}

	@Test
	public void test5() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2", "b"}, {"6", "7", "bb"}, {"11", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.FP64, Types.ValueType.STRING};
		runGenerateReaderTest(false);
	}

	// CSV with empty values
	@Test
	public void test6() {
		sampleRaw = "1,2,a,,c\n" + "6,,aa,bb,cc\n" + ",12,13,14,15";
		data = new String[][] {{"1", "2", ""}, {"6", "0", "bb"}, {"0", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest(false);
	}
}
