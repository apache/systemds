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

import com.google.gson.Gson;
import org.apache.hadoop.metrics2.sink.ganglia.GangliaSink30;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.functions.iogen.objects.ComplexObject1;
import org.apache.sysds.test.functions.iogen.objects.NumericObject1;
import org.junit.Test;

import java.util.ArrayList;

public class FrameGenerateReaderJSONTest extends GenerateReaderFrameTest {

	private final static String TEST_NAME = "MatrixGenerateReaderCSVTest";

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Test
	public void test1() {
		generateAndRun(10);
	}

	@Test
	public void test2() {
		generateAndRun(50);
	}

	@Test
	public void test3() {
		generateAndRun(100);
	}

	@Test
	public void test4() {
		generateAndRun(500);
	}

	@Test
	public void test5() {
		generateAndRun(1000);
	}

	@Test
	public void test6() {
		generateAndRun(2000);
	}

	private void generateAndRun(int nrows) {
		ComplexObject1 ot = new ComplexObject1();
		ArrayList<Object> olt = ot.getJSONFlatValues();
		int ncols = olt.size();
		names = new String[ncols];
		for(int i=0;i<ncols;i++)
			names[i]="col_"+i;
		schema = ot.getSchema().toArray(new Types.ValueType[0]);
		data = new String[nrows][ncols];
		StringBuilder sb = new StringBuilder();
		for(int r = 0; r < nrows; r++) {
			ComplexObject1 o = new ComplexObject1();
			ArrayList<Object> ol = o.getJSONFlatValues();
			int index = 0;
			for(Object oi : ol) {
				if(oi != null)
					data[r][index++] = UtilFunctions.objectToString(oi);
				else {
					Types.ValueType vt = schema[index];
					String na = "";
					switch(vt) {
						case FP32:
						case FP64:
						case INT32:
						case INT64:
							na = "0";
							break;
						case STRING:
							na = "";
							break;
						case BOOLEAN:
							na = "false";
							break;
					}
					data[r][index++] = na;
				}

			}
			sb.append(o.getJSON());
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();
		runGenerateReaderTest();
	}
}