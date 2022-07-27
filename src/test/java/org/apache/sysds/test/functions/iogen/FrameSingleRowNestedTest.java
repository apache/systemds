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

public class FrameSingleRowNestedTest extends GenerateReaderFrameTest {

	private final static String TEST_NAME = "FrameSingleRowNestedTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	// JSON Dataset
	//1. flat object, in-order values
	@Test
	public void test1() {
		sampleRaw = "{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"e\":5}\n" +
					"{\"a\":6,\"b\":7,\"c\":8,\"d\":9,\"e\":10}\n" +
					"{\"a\":11,\"b\":12,\"c\":13,\"d\":14,\"e\":15}";

		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32};
		runGenerateReaderTest(false);
	}

	//2. flat object, out-of-order values, contain different value types
	@Test
	public void test2() {
		sampleRaw = "{\"b\":\"string\",\"a\":\"1\",\"e\":5,\"c\":3,\"d\":4}\n" +
					"{\"d\":9,\"b\":\"string2\",\"c\":8,\"a\":\"6\",\"e\":10}\n" +
					"{\"d\":14,\"a\":\"11\",\"e\":15,\"b\":\"string3\",\"c\":13}";

		data = new String[][] {{"1", "string"}, {"6", "string2"}, {"11", "string3"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest(false);
	}
	//3. nested object with unique attribute names
	@Test
	public void test3() {
		sampleRaw = "{\"a\":1,\"b\":{\"c\":2,\"d\":3,\"e\":4},\"f\":5}\n" +
					"{\"a\":6,\"b\":{\"c\":7,\"d\":8,\"e\":9},\"f\":10}\n" +
					"{\"a\":11,\"b\":{\"c\":12,\"d\":13,\"e\":14},\"f\":15}\n";
		data = new String[][] {{"1", "2", "5"}, {"6", "7", "10"}, {"11", "12", "15"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING, Types.ValueType.FP64};
		runGenerateReaderTest(false);
	}

	//5. nested object with repeated attribute names, out-of-order
	@Test
	public void test5() {
		sampleRaw = "{\"a\":1,\"b\":{\"a\":2,\"b\":3,\"f\":4},\"f\":5}\n" +
					"{\"a\":6,\"b\":{\"a\":7,\"b\":8,\"f\":9},\"f\":10}\n" +
					"{\"a\":11,\"b\":{\"a\":12,\"b\":13,\"f\":14},\"f\":15}";
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING, Types.ValueType.FP64,
			Types.ValueType.FP32, Types.ValueType.INT64};
		data = new String[][] {{"1", "2", "3", "4", "5"}, {"6", "7", "8", "9", "10"}, {"11", "12", "13", "14", "15"}};
		runGenerateReaderTest(false);
	}
}
