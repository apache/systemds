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

package org.apache.sysds.test.functions.io.generateReader;

import com.google.gson.Gson;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.io.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class GenerateReaderTest extends AutomatedTestBase {

	private final static String TEST_NAME = "GenerateReaderTest";
	private final static String TEST_DIR = "functions/io/GenerateReaderTest/";
	private final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;

	private String stream;
	private MatrixBlock sample;

	@Override public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
	}

	//1. Generate CSV Test Data
	//1.a. The Data include Header and Unique Values
	@Test public void testCSV1_CP_CSV_Data_With_Header() throws IOException{
		stream = "a,b,c,d,e,f\n" +
			"1,2,3,4,5,6\n" +
			"7,8,9,10,11,12\n" +
			"2,3,1,5,4,6\n"+
			"1,5,3,4,2,6\n"+
			"8,9,10,11,12,7";

		double[][] sample = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
		MatrixReader reader = GenerateReader.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
		TestReaderCSV(reader, stream, ",", true,"testCSV1_CP_CSV_Data_With_Header");
	}
	//1.b: The Data Don't have Header and Unique Values
	@Test public void testCSV2_CP_CSV_Data_With_Header() throws IOException{
		stream = "1,2,3,4,5,6\n" +
			"7,8,9,10,11,12\n" +
			"2,3,1,5,4,6\n"+
			"1,5,3,4,2,6\n"+
			"8,9,10,11,12,7";

		double[][] sample = {{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};
		MatrixReader reader = GenerateReader.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
		TestReaderCSV(reader, stream, ",", false,"testCSV2_CP_CSV_Data_With_Header");
	}

	//1.c: The Data Header and Duplicated Values
	@Test public void testCSV3_CP_CSV_Data_With_Header() throws IOException{
		stream = "1,2,3,3,1,2\n" +
			"7,8,7,9,8,8\n" +
			"2,3,3,2,1,1\n"+
			"3,3,2,2,1,1\n"+
			"8,7,8,7,8,9";

		double[][] sample = {{1, 2, 3, 1, 2, 3}, {7, 7, 9, 8, 8, 8}};
		MatrixReader reader = GenerateReader.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
		TestReaderCSV(reader, stream, ",", false,"testCSV3_CP_CSV_Data_With_Header");
	}

	//2. Generate LIBSVM Test Data
	//2.a: The Data are Unique Values
	@Test public void testLIBSVM1_CP_LIBSVM_Data_Unique() throws IOException {
		stream = "1 1:1 2:2 6:3\n" +
				 "2 3:1 4:2 5:3\n" +
				 "1 1:2 2:3 6:1\n" +
				 "2 3:3 4:1 5:2\n" +
				 "1 1:3 2:4 6:5";
		double[][] sample = {{1,2,0,0,0,3,1}, {4,5,0,0,0,3,1},{0,0,1,2,3,0,2}};
		MatrixReader reader = GenerateReader.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
		TestReaderLIBSVM(reader, stream, " ", ":","testLIBSVM1_CP_LIBSVM_Data_Unique");
	}

	//2.a: The Data are Duplicate Values
	@Test public void testLIBSVM2_CP_LIBSVM_Data_Duplicate() throws IOException {
		stream = "1 1:1 2:2 6:3 7:3 9:1\n" +
			"2 3:1 4:2 5:3 8:5 10:5\n" +
			"1 1:2 2:3 6:1 7:2 9:1\n" +
			"2 3:3 4:1 5:2 8:2 10:5\n" +
			"1 1:3 2:4 6:5 7:3 9:5";
		double[][] sample = {{1,2,0,0,0,3,3,0,1,0,1}, {4,5,0,0,0,3,3,0,5,0,1},{0,0,1,2,3,0,0,5,0,5,2}};
		MatrixReader reader = GenerateReader.generateReader(stream, DataConverter.convertToMatrixBlock(sample));
		TestReaderLIBSVM(reader, stream, " ", ":","testLIBSVM2_CP_LIBSVM_Data_Duplicate");
	}
	private void TestReaderCSV(MatrixReader reader, String stream, String delim, boolean hasheader,String fileName) throws IOException {

		InputStream is = IOUtilFunctions.toInputStream(stream);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int rlen = 0, clen = 0;
		while((value = br.readLine()) != null) //foreach line
		{
			if(rlen ==0)
				clen = IOUtilFunctions.splitCSV(value, delim).length;
			rlen++;
		}
		if(hasheader)
			rlen--;

		is = IOUtilFunctions.toInputStream(stream);
		MatrixBlock mbStream = reader.readMatrixFromInputStream(is, rlen, clen, -1, -1);

		FileFormatPropertiesCSV format = new FileFormatPropertiesCSV(hasheader, delim, false);
		WriterTextCSV writer = new WriterTextCSV(format);
		writer.writeMatrixToHDFS(mbStream,"/home/sfathollahzadeh/GernerateReaderTests/"+fileName+".csv",rlen,clen,
			-1,-1,false);
	}


	private void TestReaderLIBSVM(MatrixReader reader, String stream, String delim, String delimIndex,String fileName) throws IOException {

		InputStream is = IOUtilFunctions.toInputStream(stream);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String value;
		int rlen = 0, clen = 0;
		FileFormatPropertiesLIBSVM format = new FileFormatPropertiesLIBSVM(delim, delimIndex);
		while((value = br.readLine()) != null) //foreach line
		{
			String items[] = IOUtilFunctions.splitCSV(value, format.getDelim());
			for(int i = 1; i < items.length; i++) {
				String cell = IOUtilFunctions.splitCSV(items[i], format.getIndexDelim())[0];
				int ci = UtilFunctions.parseToInt(cell);
				if(clen < ci) {
					clen = ci;
				}
			}
		rlen++;
		}
		clen++;
		is = IOUtilFunctions.toInputStream(stream);
		MatrixBlock mbStream = reader.readMatrixFromInputStream(is, rlen, clen, -1, -1);

		WriterTextLIBSVM writer = new WriterTextLIBSVM(format);
		writer.writeMatrixToHDFS(mbStream,"/home/sfathollahzadeh/GernerateReaderTests/"+fileName+".libsvm",rlen,clen,
			-1,-1,false);
	}
}
