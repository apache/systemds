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
		runGenerateReaderTest();
	}

	// 2. dataset contain different value types
	@Test
	public void test2() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32};
		runGenerateReaderTest();
	}

	@Test
	public void test3() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	@Test
	public void test4() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2", "b"}, {"6", "7", "bb"}, {"11", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	@Test
	public void test5() {
		sampleRaw = "1,2,a,b,c\n" + "6,7,aa,bb,cc\n" + "11,12,13,14,15";
		data = new String[][] {{"1", "2", "b"}, {"6", "7", "bb"}, {"11", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.FP64, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	// CSV with empty values
	@Test
	public void test6() {
		sampleRaw = "1,2,a,,c\n" + "6,,aa,bb,cc\n" + ",12,13,14,15";
		data = new String[][] {{"1", "2", ""}, {"6", "0", "bb"}, {"0", "12", "14"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	// LibSVM
	// with in-order col indexes and numeric col indexes
	@Test
	public void test7() {
		sampleRaw = "+1 1:10 2:20 3:30\n" + "-1 4:40 5:50 6:60\n" + "+1 1:101 2:201 \n" +
			"-1 6:601 \n" + "-1 5:501\n" + "+1 3:301";

		data = new String[][] {{"1", "10", "20", "30", "0", "", ""},
								{"-1", "0", "0", "0", "40", "50", "60"},
								{"1", "101", "201", "0", "0", "", ""},
								{"-1", "0", "0", "0", "0", "", "601"},
								{"-1", "0", "0", "0", "0", "501", ""},
								{"1", "0", "0", "301", "0", "", ""}};

		schema = new Types.ValueType[] {Types.ValueType.FP32, Types.ValueType.INT32, Types.ValueType.INT64,
										Types.ValueType.FP32, Types.ValueType.FP64, Types.ValueType.STRING, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	@Test
	public void test8() {
		sampleRaw = "+1 1:10 2:20 3:30\n" + "-1 4:40 5:a 6:b\n" + "+1 1:101 2:201 \n" +
			"-1 6:c \n" + "-1 5:d\n" + "+1 3:301";

		data = new String[][] {{"1", "10", "20", "30", "0", "", ""},
								{"-1", "0", "0", "0", "40", "a", "b"},
								{"1", "101", "201", "0", "0", "", ""},
								{"-1", "0", "0", "0", "0", "", "c"},
								{"-1", "0", "0", "0", "0", "d", ""},
								{"1", "0", "0", "301", "0", "", ""}};

		schema = new Types.ValueType[] {Types.ValueType.FP32, Types.ValueType.INT32, Types.ValueType.INT64,
			Types.ValueType.FP32, Types.ValueType.FP64, Types.ValueType.STRING, Types.ValueType.STRING};
		runGenerateReaderTest();
	}

	// MatrixMarket(MM)
	//MM with inorder dataset, (RowIndex,Col Index,Value). Row & Col begin index: (1,1)
	@Test
	public void test9() {
		sampleRaw = "1,1,10\n" + "1,2,20\n" + "1,3,30\n"+ "1,5,50\n" + "2,1,101\n" + "2,2,201\n" + "4,1,104\n" +
			"4,5,504\n" + "5,3,305";
		data = new String[][] {{"10","20","30"},
								{"101","201",""},
								{"0","0",""},
								{"104", "0", ""},
								{"0", "0", "305"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.FP64, Types.ValueType.STRING};
		runGenerateReaderTest();
	}


}
