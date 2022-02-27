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

import com.google.gson.Gson;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.io.FileFormatPropertiesLIBSVM;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderJSONJackson;
import org.apache.sysds.runtime.io.FrameReaderJSONL;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.ReaderTextLIBSVM;
import org.apache.sysds.runtime.iogen.FormatIdentifying;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.iogen.EXP.Util;
import org.apache.sysds.runtime.iogen.Hirschberg;
import org.apache.sysds.runtime.iogen.MappingTrie;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.functions.iogen.GenerateReaderMatrixTest;
import org.junit.Test;

import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

	@Test public void test101() throws IOException {

		FileFormatPropertiesLIBSVM propertiesLIBSVM = new FileFormatPropertiesLIBSVM(" ", ":");
		ReaderTextLIBSVM readerTextLIBSVM = new ReaderTextLIBSVM(propertiesLIBSVM);
		MatrixBlock mb = readerTextLIBSVM.readMatrixFromHDFS("/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/susy-libsvm/susy-libsvm.data",-1,18,-1,-1);
	}

	@Test public void test13() throws Exception {
		///home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/twitter-examples/F10
		for(int f = 1; f <= 2; f++) {
			System.out.println("+++++++++++++++++++++  Q=" + f);
			String sampleRawFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/HIGGS-csv/Q" + f + "/sample-HIGGS-csv200.raw";
			String sampleFrameFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/HIGGS-csv/Q" + f + "/sample-HIGGS-csv200.frame";
			String delimiter = "\\t";
			String schemaFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/HIGGS-csv/Q" + f + "/HIGGS-csv.schema";
			String dataFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/HIGGS-csv/HIGGS-csv.data";

			Util util = new Util();
			Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
			int ncols = sampleSchema.length;

			ArrayList<Types.ValueType> newSampleSchema = new ArrayList<>();
			ArrayList<ArrayList<String>> newSampleFrame = new ArrayList<>();

			String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, delimiter,0);

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
			FrameReader fr = gr.getReader();

			FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, sampleSchema.length);
			int a = 100;

		}
	}



	@Test public void test14() throws Exception {
		///home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/twitter-examples/F10
		for(int f = 1; f <= 784; f++) {
			System.out.println("+++++++++++++++++++++  Q=" + f);
			String sampleRawFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/mnist8m-libsvm/F" + f + "/sample-mnist8m-libsvm200.raw";
			String sampleMatrixFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/mnist8m-libsvm/F" + f + "/sample-mnist8m-libsvm200.matrix";
			String delimiter = "\\t";
			String dataFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/mnist8m-libsvm/mnist8m-libsvm.data";

			Util util = new Util();

			MatrixBlock sampleMB = util.loadMatrixData(sampleMatrixFileName, delimiter);
			String sampleRaw = util.readEntireTextFile(sampleRawFileName);

			GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB);
			MatrixReader mr = gr.getReader();
//			MatrixBlock matrixBlock = mr.readMatrixFromHDFS(dataFileName, -1, sampleMB.getNumColumns(), -1, -1);

//			FormatIdentifying fi = new FormatIdentifying(sampleRaw,sampleMB);
//
//			myregex mr = new myregex(fi.getFormatProperties());
//			mr.readMatrixFromHDFS(dataFileName, -1, sampleMB.getNumColumns(), -1, -1);

			int a = 100;

		}
	}

	@Test public void test15() throws Exception {
		String str = "0 1:9.728614687919616699e-01 2:6.538545489311218262e-01 3:1.176224589347839355e+00 4:1.157156467437744141e+00 5:-1.739873170852661133e+00 6:-8.743090629577636719e-01 7:5.677649974822998047e-01 8:-1.750000417232513428e-01 9:8.100607395172119141e-01 10:-2.525521218776702881e-01 11:1.921887040138244629e+00 12:8.896374106407165527e-01 13:4.107718467712402344e-01 14:1.145620822906494141e+00 15:1.932632088661193848e+00 16:9.944640994071960449e-01 17:1.367815494537353516e+00 18:4.071449860930442810e-02";
		String str1="0 1:0.30151 2:0.30151 3:0.30151 4:0.30151 5:0.30151 6:0.30151 7:0.30151 8:0.30151 9:0.30151 10:0.30151 11:0.30151";


//		String str = "     123:";
//		String s= str.replaceAll("\\d+","\\\\d+");
//		System.out.println(s);

		//(?<=^|[\w\d]\s)([\w\d]+)(?=\s|$)

		String regex = "(\\d+:)";//"(?<=\\d:)(.*?)(?=\\d:)"; //(.*?)(\d+:)

//		String regex="\\d+:";

		List<String> allMatches = new ArrayList<String>();

		for(int i=0;i<10000000;i++) {
			Matcher m = Pattern.compile(regex).matcher(str1);
			while(m.find()) {
				String s = m.group(1) + "  ";//+ m.group(3);//+"  "+ m.group(5);
				//System.out.println(s);
				//allMatches.add(m.group(5));
			}
		}



//
//		Pattern p = Pattern.compile(regex);
//
//		// Find match between given string
//		// and regular expression
//		// using Pattern.matcher()
//		Matcher m = p.matcher(str);
//
//		// Get the subsequence
//		// using find() method
//		while(m.find()) {
//			System.out.println(m.group()+"  "+m.start()+" "+ m.end()+"  ");
//		}

		//		int misMatchPenalty = 3;
		//		int gapPenalty = 2;
		//		Hirschberg hirschberg = new Hirschberg();

		//		ArrayList<String> list = new ArrayList<>();
		//		for(int i=0;i<100000000;i++){
		//			list.add(" "+i+":"+i+"--");
		//		}
		//
		//		ArrayList<String> ll = hirschberg.getLCS(list, misMatchPenalty,gapPenalty);
		//		Gson gson = new Gson();
		//		System.out.println(gson.toJson(ll));
		//
		//
		//
		////		List<String> allMatches = new ArrayList<String>();
		////		Matcher m = Pattern.compile("\\s\\w:").matcher(str);
		////		while (m.find()) {
		////
		////			allMatches.add(m.group());
		////		}
		////		for(String s: allMatches)
		////			System.out.println(s);
		////

		//---------------------------------------------
		// Regex to extract the string
		// between two delimiters
		//		String regex = "\\[(.*?)\\]";
		//
		//		// Compile the Regex.
		//		Pattern p = Pattern.compile(regex);
		//
		//		// Find match between given string
		//		// and regular expression
		//		// using Pattern.matcher()
		//		Matcher m = p.matcher(str);
		//
		//		// Get the subsequence
		//		// using find() method
		//		while (m.find())
		//		{
		//			System.out.println(m.group(1));
		//		}
		//		//----------------------------------------------
		//		Pattern.compile()
		//		MappingTrie mappingTrie = new MappingTrie();
		//		for(int i=0;i<1000000;i++){
		//			mappingTrie.insert(" "+i+":",i);
		//		}
		//
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 123:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 124:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 125:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 256233:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 58296:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 10000:",0);
		//		mappingTrie.insert(","+Lop.OPERAND_DELIMITOR+" 9658263:",0);
		//
		//		boolean flag=false;
		//		do {
		//			flag = mappingTrie.reConstruct();
		//		}while(flag);
		//
		//		ArrayList<ArrayList<String>> myList = mappingTrie.getAllSequentialKeys();
		//		Gson gson = new Gson();
		//		System.out.println(gson.toJson(myList.get(0)));

	}


	@Test public void test16() throws Exception {
		///home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/twitter-examples/F10
		for(int f = 1; f <= 2; f++) {
			System.out.println("+++++++++++++++++++++  Q=" + f);
			String sampleRawFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/Q" + f + "/sample-yelp-csv200.raw";
			String sampleFrameFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/Q" + f + "/sample-yelp-csv200.frame";
			String delimiter = "\\t";
			String dataFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/yelp-csv.data";
			String schemaFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/yelp-csv/Q" + f + "/yelp-csv.schema";

			Util util = new Util();
			Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
			int ncols = sampleSchema.length;

			String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, delimiter,ncols);

			FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);
			String sampleRaw = util.readEntireTextFile(sampleRawFileName);

			GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame);
			FrameReader fr = gr.getReader();

			//FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, -1, ncols);
			int a = 100;
		}
	}



	@Test public void test17() throws Exception {

		MatrixBlock m = new MatrixBlock(10,10,true);

		for(int f = 2; f <= 2; f++) {
			System.out.println("+++++++++++++++++++++  Q=" + f);
			String sampleRawFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/queen-mm/F" + f + "/sample-queen" +
					"-mm200.raw";
			String sampleMatrixFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/queen-mm/F" + f + "/sample-queen-mm200.matrix";
			String delimiter = "\\t";
			String dataFileName = "/home/saeed/Documents/Github/papers/2022-vldb-GIO/Experiments/data/queen-mm/queen-mm.data";

			Util util = new Util();

			MatrixBlock sampleMB = util.loadMatrixData(sampleMatrixFileName, delimiter);
			String sampleRaw = util.readEntireTextFile(sampleRawFileName);

			GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB);
			MatrixReader mr = gr.getReader();
			//			MatrixBlock matrixBlock = mr.readMatrixFromHDFS(dataFileName, -1, sampleMB.getNumColumns(), -1, -1);

			//			FormatIdentifying fi = new FormatIdentifying(sampleRaw,sampleMB);
			//
			//			myregex mr = new myregex(fi.getFormatProperties());
			//			mr.readMatrixFromHDFS(dataFileName, -1, sampleMB.getNumColumns(), -1, -1);

			int a = 100;

		}
	}
}
