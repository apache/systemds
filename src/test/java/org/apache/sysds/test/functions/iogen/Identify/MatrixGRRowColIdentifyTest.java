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

package org.apache.sysds.test.functions.iogen.Identify;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderJSONJackson;
import org.apache.sysds.runtime.io.FrameReaderJSONL;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.iogen.EXP.Util;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.test.functions.iogen.GenerateReaderMatrixTest;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;

public class MatrixGRRowColIdentifyTest extends GenerateReaderMatrixTest {

	private final static String TEST_NAME = "MatrixGenerateReaderCSVTest";

	@Override protected String getTestName() {
		return TEST_NAME;
	}

	private void generateRandomCSV(int nrows, int ncols, double min, double max, double sparsity, String separator,
		String[] naString) {

		sampleMatrix = getRandomMatrix(nrows, ncols, min, max, sparsity, 714);
		StringBuilder sb = new StringBuilder();

		for(int r = 0; r < nrows; r++) {
			StringBuilder row = new StringBuilder();
			for(int c = 0; c < ncols; c++) {
				if(sampleMatrix[r][c] != 0) {
					row.append(sampleMatrix[r][c]).append(separator);
				}
				else {
					Random rn = new Random();
					int rni = rn.nextInt(naString.length);
					row.append(naString[rni]).append(separator);
				}
			}

			sb.append(row.substring(0, row.length() - separator.length()));
			if(r != nrows - 1)
				sb.append("\n");
		}
		sampleRaw = sb.toString();
	}

	@Test public void test1() {
		sampleRaw = "1,2,3,4,5\n" + "6,7,8,9,10\n" + "11,12,13,14,15";
		sampleMatrix = new double[][] {{1, 2}, {6, 7}, {11, 12}};
		runGenerateReaderTest();
	}

	@Test public void test2() {
		sampleRaw = "1,a2,a3,a4,a5\n" + "6,a7,a8,a9,a10\n" + "11,a12,a13,a14,a15";
		sampleMatrix = new double[][] {{1, 5}, {6, 10}, {11, 15}};
		runGenerateReaderTest();
	}

	@Test public void test3() {
		sampleRaw = "1,,2,,3,,4,,5\n" + "6,,7,,8,,9,,10\n" + "11,,12,,13,,14,,15";
		sampleMatrix = new double[][] {{1, 5}, {6, 10}, {11, 15}};
		runGenerateReaderTest();
	}

	@Test public void test4() {
		String[] naString = {"NaN"};
		generateRandomCSV(20, 20, -10, 10, 1, ",", naString);
		runGenerateReaderTest();
	}

	@Test public void test5() {
		sampleRaw = "{\"name\":1, \"occupation\":2, \"user\":{\"name\":3,\"password\":4}}\n" + "{\"name\":6, \"occupation\":7, \"user\":{\"name\":8,\"password\":9}}\n" + "{\"name\":10, \"occupation\":11, \"user\":{\"name\":12,\"password\":13}}\n" + "{\"name\":14, \"occupation\":15, \"user\":{\"name\":16,\"password\":17}}\n" + "{\"name\":18, \"occupation\":19, \"user\":{\"name\":20,\"password\":21}}";
		sampleMatrix = new double[][] {{2, 3}, {7, 8}, {11, 12}, {15, 16}, {19, 20}};
		runGenerateReaderTest();
	}

	@Test public void test6() {
		sampleRaw = "{\"name\":1, \"occupation\":2, \"user\":{\"password\":4, \"name\":3}}\n" + "{\"name\":6, \"occupation\":7, \"user\":{\"name\":8,\"password\":9}}\n" + "{\"name\":10, \"occupation\":11, \"user\":{\"name\":12,\"password\":13}}\n" + "{\"name\":14, \"occupation\":15, \"user\":{\"name\":16,\"password\":17}}\n" + "{\"name\":18, \"occupation\":19, \"user\":{\"name\":20,\"password\":21}}";
		sampleMatrix = new double[][] {{2, 3}, {7, 8}, {11, 12}, {15, 16}, {19, 20}};
		runGenerateReaderTest();
	}

	@Test public void test7() {
		sampleRaw = "{\"name\":1, \"occupation\":2, \"user\":{\"password\":4, \"name\":3}}\n" + "{\"name\":6, \"occupation\":7, \"user\":{\"name\":8,\"password\":9}}\n" + "{\"name\":10, \"occupation\":11, \"user\":{\"name\":12,\"password\":13}}\n" + "{\"name\":14, \"occupation\":15, \"user\":{\"name\":16,\"password\":17}}\n" + "{\"name\":18, \"user\":{\"name\":20,\"password\":21}, \"occupation\":19}";
		sampleMatrix = new double[][] {{2, 3}, {7, 8}, {11, 12}, {15, 16}, {19, 20}};
		runGenerateReaderTest();
	}

	@Test public void test8() {
		sampleRaw = "1,1,10\n" + "1,2,20\n" + "1,3,30\n" + "2,1,40\n" + "2,2,50\n" + "2,3,60\n" + "3,1,70\n" + "3,2,80\n" + "3,3,90\n";

		sampleMatrix = new double[][] {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
		runGenerateReaderTest();
	}

	@Test public void test9() {
		sampleRaw = "<article>\n" + // 0
			"<author>1</author>\n" + //1
			"<author>2</author>\n" + // 2
			"<author>3</author>\n" + // 3
			"<year>1980</year>\n" + // 4
			"<title>GIO</title>\n" + // 5
			"</article>\n" + // 6
			"<article>\n" + // 7
			"<author>10</author>\n" + // 8
			"<author>20</author>\n" + // 9
			"<author>30</author>\n" + // 10
			"<year>2000</year>\n" + // 11
			"<title>GIO2</title>\n" + // 12
			"</article>\n" + // 13
			"<article>\n" + // 14
			"<year>2010</year>\n" + // 15
			"<author>100</author>\n" + // 16
			"<author>200</author>\n" + // 17
			"<author>300</author>\n" + // 18
			"<author>800</author>\n" + // 18
			"<title>GIO3</title>\n" + // 19
			"</article>\n" + // 20
			"<article>\n" + // 21
			"<author>1000</author>\n" + // 22
			"<author>2000</author>\n" + // 23
			"<author>3000</author>\n" + // 24
			"<year>2222</year>\n" + // 25
			"<title>GIO4</title>\n" + // 26
			"</article>"; // 27

		sampleMatrix = new double[][] {{1, 2, 3, 1980}, {10, 20, 30, 2000}, {100, 200, 300, 2010},
			{1000, 2000, 3000, 2222}};
		runGenerateReaderTest();
	}

	@Test public void test10() {
		sampleRaw = "<article> \n" + "<year>1980</year> \n" + "<author>1</author> \n" + "<author>2</author> \n" + "<author>3</author> \n" + "<title>GIO</title> \n" + "</article>\n" + "<book> \n" + "<author>10</author> \n" + "<author>21</author> \n" + "<author>30</author> \n" + "<year>2000</year> \n" + "<title>GIO2</title> \n" + "</book>\n" + "<homepage> \n" + "<author>100</author> \n" + "<author>300</author> \n" + "<year>210</year> \n" + "<title>GIO3</title> \n" + "<author>200</author> \n" + "</homepage>\n" + "<article> \n" + "<year>2222</year> \n" + "<author>1000</author> \n" + "<author>2000</author> \n" + "<author>3000</author> \n" + "<title>GIO4</title> \n" + "</article>";

		sampleMatrix = new double[][] {{1, 2, 3, 1980}, {10, 21, 30, 2000}, {100, 200, 300, 2010},
			{1000, 2000, 3000, 2222}};
		runGenerateReaderTest();
	}

	@Test public void test11() {
		sampleRaw = "#index 1\n" + "#t 2,3\n" + "#s 1980\n" + "#index 10\n\n" + "#t 21,30\n" + "#s 2000\n\n" + "#index 100\n" + "#t 200,300\n" + "#s 2222";

		sampleMatrix = new double[][] {{1, 2, 3, 1980}, {10, 21, 30, 2000}, {100, 200, 300, 2010},
			{1000, 2000, 3000, 2222}};
		runGenerateReaderTest();
	}

	@Test public void test12() {
		//		sampleRaw = "#index 1\n" +
		//			"#t 2,3\n" +
		//			"#s 1980\n"+
		//			"#index 10\n\n" +
		//			"#t 21,30\n" +
		//			"#s 2000\n\n"+
		//			"#index 100\n" +
		//			"#t 200,300\n" +
		//			"#s 2222";
		//
		//		sampleMatrix = new double[][] {{1,2,3}, {10,21,30}, {100,200,300},{1000,2000,3000}};
		//		runGenerateReaderTest();

		StringBuilder sb = new StringBuilder(
			" ,)R2I( hcraeseR mmocofnI rof etutitsnI ,tnemtrapeD gniniM ataD\"[:\"snoitailiffa\",\"tuhN hniM neyugN \":\"eman\",802:\"xedni\"{");
		System.out.println(sb.reverse());
	}

	//	@Test
	//	public void test13() throws Exception {
	//		String sampleRawFileName = "/home/saeed/Documents/Dataset/GIODataset/aminer/csv/Q2/sample_aminer_author1000_5.raw";
	//		String sampleFrameFileName = "/home/saeed/Documents/Dataset/GIODataset/aminer/csv/Q2/sample_aminer_author1000_5.frame";
	//		Integer sampleNRows = 1000;
	//		String delimiter = "\\t";
	//		String schemaFileName = "/home/saeed/Documents/Dataset/GIODataset/aminer/csv/Q2/aminer_author_5.schema";
	//		String dataFileName = "/home/saeed/Documents/Dataset/GIODataset/aminer/csv/aminer_author.data";
	//
	//		Float percent = 7f;//Float.parseFloat(args[6]);
	//		String datasetName = "aminer_paper";//args[7];
	//		String LOG_HOME ="/home/saeed/Documents/ExpLog";//args[8];
	//
	//		if(delimiter.equals("\\t"))
	//			delimiter = "\t";
	//
	//		Util util = new Util();
	//		Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
	//		int ncols = sampleSchema.length;
	//
	//		ArrayList<Types.ValueType> newSampleSchema = new ArrayList<>();
	//		ArrayList<ArrayList<String>> newSampleFrame = new ArrayList<>();
	//
	//		String[][] sampleFrameStrings =  util.loadFrameData(sampleFrameFileName, sampleNRows, ncols, delimiter);
	//
	//		for(int c = 0; c < sampleFrameStrings[0].length; c++) {
	//			HashSet<String> valueSet = new HashSet<>();
	//			for(int r=0; r<sampleFrameStrings.length;r++)
	//				valueSet.add(sampleFrameStrings[r][c]);
	//			if(valueSet.size()>0){
	//				ArrayList<String> tempList = new ArrayList<>();
	//				for(int r=0; r<sampleFrameStrings.length;r++) {
	//					tempList.add(sampleFrameStrings[r][c]);
	//				}
	//				newSampleFrame.add(tempList);
	//				newSampleSchema.add(sampleSchema[c]);
	//			}
	//		}
	//
	//		sampleFrameStrings = new String[newSampleFrame.get(0).size()][newSampleFrame.size()];
	//
	//		for(int row=0; row<sampleFrameStrings.length; row++){
	//			for(int col=0; col<sampleFrameStrings[0].length; col++){
	//				sampleFrameStrings[row][col] = newSampleFrame.get(col).get(row);
	//			}
	//		}
	//
	//		sampleSchema = new Types.ValueType[newSampleSchema.size()];
	//		for(int i=0; i< newSampleSchema.size();i++)
	//			sampleSchema[i] = newSampleSchema.get(i);
	//
	//		FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);
	//
	//		double tmpTime = System.nanoTime();
	//		String sampleRaw = util.readEntireTextFile(sampleRawFileName);
	//		GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
	//		FrameReader fr = gr.getReader();
	//		double generateTime = (System.nanoTime() - tmpTime) / 1000000000.0;
	//
	//		tmpTime = System.nanoTime();
	//		FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, sampleSchema.length);
	//		double readTime = (System.nanoTime() - tmpTime) / 1000000000.0;
	//
	//		String log= datasetName+","+ frameBlock.getNumRows()+","+ ncols+","+percent+","+ sampleNRows+","+ generateTime+","+readTime;
	//		util.addLog(LOG_HOME, log);
	//	}

	@Test public void test13() throws Exception {
		///home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/twitter-examples/F10
		for(int f=9;f<=9;f++) {
			System.out.println("+++++++++++++++++++++  Q="+f);
			String sampleRawFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/F"+f+"/sample-yelp-csv200.raw";
			String sampleFrameFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/F"+f+"/sample-yelp-csv200.frame";
			String delimiter = "\\t";
			String schemaFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/F"+f+"/yelp-csv.schema";
			String dataFileName ="/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/yelp-csv.data";

			Util util = new Util();
			Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
			int ncols = sampleSchema.length;

			ArrayList<Types.ValueType> newSampleSchema = new ArrayList<>();
			ArrayList<ArrayList<String>> newSampleFrame = new ArrayList<>();

			String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, ncols, delimiter);

			for(int c = 0; c < sampleFrameStrings[0].length; c++) {
				HashSet<String> valueSet = new HashSet<>();
				for(int r = 0; r < sampleFrameStrings.length; r++)
					valueSet.add(sampleFrameStrings[r][c]);
				if(valueSet.size() > 1) {
					ArrayList<String> tempList = new ArrayList<>();
					for(int r = 0; r < sampleFrameStrings.length; r++) {
						tempList.add(sampleFrameStrings[r][c]);
					}
					newSampleFrame.add(tempList);
					newSampleSchema.add(sampleSchema[c]);
				}
			}

			sampleFrameStrings = new String[newSampleFrame.get(0).size()][newSampleFrame.size()];

			for(int row = 0; row < sampleFrameStrings.length; row++) {
				for(int col = 0; col < sampleFrameStrings[0].length; col++) {
					sampleFrameStrings[row][col] = newSampleFrame.get(col).get(row);
				}
			}

			sampleSchema = new Types.ValueType[newSampleSchema.size()];
			for(int i = 0; i < newSampleSchema.size(); i++)
				sampleSchema[i] = newSampleSchema.get(i);

			//String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, ncols, delimiter);

			FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);

			String sampleRaw = util.readEntireTextFile(sampleRawFileName);

			GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
			FrameReader fr =gr.getReader();

			FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, sampleSchema.length);

		}
	}

	@Test public void test14() throws Exception {
//		FrameReaderJSONL frameReaderJSONL = new FrameReaderJSONL();
//
//		String FILENAME_SINGLE = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/tpch-json/Q3/sample-tpch-json200.raw";
//		Types.ValueType[] schema = {Types.ValueType.STRING,Types.ValueType.STRING,Types.ValueType.FP64,Types.ValueType.FP64,Types.ValueType.FP64,Types.ValueType.FP64};
//
//		Map<String, Integer> schemaMap = new HashMap<>();
//		schemaMap.put("/returnFlag",0);
//		schemaMap.put("/lineStatus",1);
//		schemaMap.put("/quantity",2);
//		schemaMap.put("/extendedPrice",3);
//		schemaMap.put("/discount",4);
//		schemaMap.put("/tax",5);
//		// Read FrameBlock
//		FrameBlock readBlock = frameReaderJSONL.readFrameFromHDFS(FILENAME_SINGLE, schema, schemaMap, -1, schema.length);
//
//		int a = 100;

		String schemaFileName ="/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/twitter-json/F10/twitter-json.schema";
		String schemaMapFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/twitter-json/F10/twitter-json.schemaMap";
		String dataFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/twitter-json/twitter-json.data";
		long nrows = 1000;

		Util util = new Util();
		Types.ValueType[] schema = util.getSchema(schemaFileName);
		int ncols = schema.length;
		Map<String, Integer> schemaMap = util.getSchemaMap(schemaMapFileName);

		FrameReaderJSONJackson frameReaderJSONJackson = new FrameReaderJSONJackson();
		FrameBlock readBlock = frameReaderJSONJackson.readFrameFromHDFS(dataFileName, schema, schemaMap, nrows, ncols);
	}
}
