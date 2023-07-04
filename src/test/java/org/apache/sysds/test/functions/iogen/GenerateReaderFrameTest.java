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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public abstract class GenerateReaderFrameTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderFrameTest.class.getSimpleName() + "/";
	protected String sampleRaw;
	protected String[][] data;
	protected String[] names;
	protected Types.ValueType[] schema;
	protected Types.ValueType[] types= {
		Types.ValueType.STRING,
		Types.ValueType.INT32,
		Types.ValueType.INT64,
		Types.ValueType.FP32,
		Types.ValueType.FP64,
		Types.ValueType.BOOLEAN};

	protected Types.ValueType[] types1= { Types.ValueType.BOOLEAN};

	protected abstract String getTestName();

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(TEST_DIR, getTestName(), new String[] {"Y"}));
	}

	protected String getRandomString(int length) {
		//String alphabet1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890";
		String alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		StringBuilder salt = new StringBuilder();
		Random rnd = new Random();
		while (salt.length() < length) { // length of the random string.
			int index = (int) (rnd.nextFloat() * alphabet.length());
			salt.append(alphabet.charAt(index));
		}
		String saltStr = salt.toString();
		return saltStr;
	}

	@SuppressWarnings("incomplete-switch")
	protected String defaultValue(Types.ValueType vt){
		switch(vt){
			case STRING: return "";
			case BOOLEAN: return null;
			case FP32:
			case FP64:
			case INT32:
			case INT64:
				return "0";
		}
		return null;
	}

	protected void generateRandomString(int size, int maxStringLength, String[] naStrings, double sparsity, String[][] data, int colIndex) {

		double[][] lengths = getRandomMatrix(size, 1, 10, maxStringLength, sparsity, 714);

		for(int i = 0; i < size; i++) {
			int length = (int) lengths[i][0];
			if(length > 0) {
				String generatedString = getRandomString(length);
				data[i][colIndex] = generatedString;
			}
			else {
				data[i][colIndex] = null;
			}
		}
	}

	@SuppressWarnings("incomplete-switch")
	protected void generateRandomNumeric(int size, Types.ValueType type, double min, double max, String[] naStrings,
		double sparsity, String[][] data, int colIndex) {

		double[][] randomData = getRandomMatrix(size, 1, min, max, sparsity, -1);
		for(int i = 0; i < size; i++) {
			if(randomData[i][0] != 0) {
				Object o = null;
				switch(type){
					case INT32: o = UtilFunctions.objectToObject(type,(int)randomData[i][0]); break;
					case INT64: o = UtilFunctions.objectToObject(type,(long)randomData[i][0]); break;
					case FP32: o = UtilFunctions.objectToObject(type,(float)randomData[i][0]); break;
					case FP64: o = UtilFunctions.objectToObject(type,randomData[i][0]); break;
					case BOOLEAN: Boolean b= randomData[i][0] >0 ? true: null; o = UtilFunctions.objectToObject(type, b); break;
				}
				String s = UtilFunctions.objectToString(o);
				data[i][colIndex] = s;
			}
			else {
				if(type.isNumeric())
					data[i][colIndex] ="0";
				else
					data[i][colIndex] =null;
			}
		}
	}

	protected void generateRandomData(int nrows, int ncols, double min, double max, double sparsity, String[] naStrings) {

		names = new String[ncols];
		schema = new Types.ValueType[ncols];
		data = new String[nrows][ncols];

		for(int i = 0; i < ncols; i++) {
			names[i] = "C_" + i;

			Random rn = new Random();
			int rnt = rn.nextInt(types.length);
			if(i == 0|| i==ncols-1)
				rnt = 3;
			schema[i] = types[rnt];

			if(types[rnt] == Types.ValueType.STRING)
				generateRandomString(nrows,100,naStrings,sparsity,data,i);
			else if(types[rnt].isNumeric() || types[rnt] == Types.ValueType.BOOLEAN)
				generateRandomNumeric(nrows, types[rnt],min,max,naStrings, sparsity,data,i);
			}
	}

	protected void runGenerateReaderTest() {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = false;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);

			FrameBlock sampleFrame = new FrameBlock(schema, names, data);

			String HOME = SCRIPT_DIR + TEST_DIR;
			File directory = new File(HOME);
			if (! directory.exists()){
				directory.mkdir();
			}
			String dataPath = HOME + "frame_data.raw";
			int clen = data[0].length;
			writeRawString(sampleRaw, dataPath);
			GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);

			FrameReader fr= gr.getReader();
			fr.readFrameFromHDFS(dataPath,schema,names,data.length, clen);
		}
		catch(Exception exception) {
			exception.printStackTrace();
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private static void writeRawString(String raw, String fileName) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		writer.write(raw);
		writer.close();
	}
}
