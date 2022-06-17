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
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.EXP.Util;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.wink.json4j.JSONObject;
import org.junit.Test;

import java.io.IOException;

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
		runGenerateReaderTest();
	}

	//2. flat object, out-of-order values, contain different value types
	@Test
	public void test2() {
		sampleRaw = "{\"b\":\"string\",\"a\":\"1\",\"e\":5,\"c\":3,\"d\":4}\n" +
					"{\"d\":9,\"b\":\"string2\",\"c\":8,\"a\":\"6\",\"e\":10}\n" +
					"{\"d\":14,\"a\":\"11\",\"e\":15,\"b\":\"string3\",\"c\":13}";

		data = new String[][] {{"1", "string"}, {"6", "string2"}, {"11", "string3"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING};
		runGenerateReaderTest();
	}
	//3. nested object with unique attribute names
	@Test
	public void test3() {
		sampleRaw = "{\"a\":1,\"b\":{\"c\":2,\"d\":3,\"e\":4},\"f\":5}\n" +
					"{\"a\":6,\"b\":{\"c\":7,\"d\":8,\"e\":9},\"f\":10}\n" +
					"{\"a\":11,\"b\":{\"c\":12,\"d\":13,\"e\":14},\"f\":15}\n";
		data = new String[][] {{"1", "2", "5"}, {"6", "7", "10"}, {"11", "12", "15"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING, Types.ValueType.FP64};
		runGenerateReaderTest();
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
		runGenerateReaderTest();
	}

	@Test
	public void test6() {
		sampleRaw = "{\"index\":207,\"name\":\"Nuno Guimarães\",\"affiliations\":[\"ISCTEUniversity Institute of Lisbon, Lisbon, Portugal\"],\"paperCount\":1,\"citationNumber\":0,\"hIndex\":0.0,\"researchInterests\":[\"mental state\",\"mental workload\",\"higher mental workload\",\"mental load\",\"mental workload evaluation\",\"mental workload pattern\",\"ecological reading situation\",\"reading condition\",\"visual user interface\",\"EEG signal\"]}\n"+
		"{\"index\":208,\"name\":\" Nguyen Minh Nhut\",\"affiliations\":[\"Data Mining Department, Institute for Infocomm Research (I2R), 1 Fusionopolis Way, Connexis (South Tower), Singapore 138632\"],\"paperCount\":1,\"citationNumber\":0,\"hIndex\":0.0,\"researchInterests\":[\"system health monitoring\",\"sensor node\",\"adaptive classification system architecture\",\"effective health monitoring system\",\"proposed system\",\"real-time adaptive classification system\",\"adaptive sampling frequency\",\"different sampling\",\"different sampling rate\",\"individual sensor\"]}\n\n"+
		"{\"index\":209,\"name\":\"Louis Janus\",\"affiliations\":[\"\"],\"paperCount\":1,\"citationNumber\":0,\"hIndex\":0.0,\"researchInterests\":[\"language instruction\"]}";
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.STRING, Types.ValueType.FP64,
			Types.ValueType.FP32, Types.ValueType.INT64};
		data = new String[][] {{"1", "2", "3", "4", "5"}, {"6", "7", "8", "9", "10"}, {"11", "12", "13", "14", "15"}};
		runGenerateReaderTest();
	}

	@Test
	public void test7() {
		sampleRaw = "{\n\"a\":1,\n\"b\":2,\n\"c\":3,\n\"d\":4,\n\"e\":5\n}\n" +
			"{\"a\":6,\n\"b\":7,\"c\":8,\"d\":9,\"e\":10\n}\n" +
			"{\"a\":11,\"b\":12,\n\"c\":13,\"d\":14,\"e\":15\n}";

		data = new String[][] {{"1", "2"}, {"6", "7"}, {"11", "12"}};
		schema = new Types.ValueType[] {Types.ValueType.INT32, Types.ValueType.INT32};
		runGenerateReaderTest();
	}

	@Test
	public void test8() throws Exception {
		//java -Xms15g -Xmx15g  -Dparallel=true -cp ./lib/*:./SystemDS.jar org.apache.sysds.runtime.iogen.EXP.GIOFrame
		String dpath = "/home/sfathollahzadeh/Documents/GitHub/papers/2022-vldb-GIO/Experiments/";
		String sampleRawFileName = dpath+"data/aminer-author-json/Q4/sample-aminer-author-json200.raw";
		String sampleFrameFileName = dpath+"data/aminer-author-json/Q4/sample-aminer-author-json200.frame";
		String sampleRawDelimiter = "\t";
		String schemaFileName = dpath+"data/aminer-author-json/Q4/aminer-author-json.schema";
		String dataFileName = dpath+"data/aminer-author-json.dat";
		boolean parallel = false;
		long rows = -1;
		Util util = new Util();

		// read and parse mtd file
		String mtdFileName = dataFileName + ".mtd";
		try {
			String mtd = util.readEntireTextFile(mtdFileName);
			mtd = mtd.replace("\n", "").replace("\r", "");
			mtd = mtd.toLowerCase().trim();
			JSONObject jsonObject = new JSONObject(mtd);
			if (jsonObject.containsKey("rows")) rows = jsonObject.getLong("rows");
		} catch (Exception exception) {}

		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
		int ncols = sampleSchema.length;

		String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, sampleRawDelimiter, ncols);
		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);
		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
		GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame, parallel);
		FrameReader fr = gr.getReader();
		FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, rows, sampleSchema.length);

		int a = 100;
	}
}
