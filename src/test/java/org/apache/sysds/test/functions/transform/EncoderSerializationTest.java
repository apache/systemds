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

package org.apache.sysds.test.functions.transform;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderBin;
import org.apache.sysds.runtime.transform.encode.EncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderDummycode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderPassThrough;
import org.apache.sysds.runtime.transform.encode.EncoderRecode;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.Statistics;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.junit.Assert;
import org.junit.Test;

public class EncoderSerializationTest extends AutomatedTestBase
{
//	private final static String TEST_NAME1 = "TransformFrameEncodeDecode";
	private final static String TEST_DIR = "functions/transform/";
//	private final static String TEST_CLASS_DIR = TEST_DIR + FrameEncodeDecodeSerTest.class.getSimpleName() + "/";
	String HOME = SCRIPT_DIR + TEST_DIR;
	private final static String OUTPUT_FILE = "file1.txt";

	//dataset and transform tasks without missing values
	private final static String DATASET 	= "homes/homes2.csv";
//	private final static String SPEC1 		= "homes3/homes.tfspec_recode.json";
//	private final static String SPEC1b 		= "homes3/homes.tfspec_recode2.json";
//	private final static String SPEC2 		= "homes3/homes.tfspec_dummy.json";
//	private final static String SPEC2b 		= "homes3/homes.tfspec_dummy2.json";
	
	public enum TransformType {
		RECODE,
		DUMMY,
		BIN,
		IMPUTE,
		OMIT,
		COMPOSITE,
		HASH,
		PASS
	}

	@Override
	public void setUp()  { TestUtils.clearAssertionInformation(); }

	@Test
	public void testDummy() {
		runTransformSerTest(TransformType.DUMMY);
	}

	@Test
	public void testRecode() { runTransformSerTest(TransformType.RECODE); }

	@Test
	public void testComposite() { runTransformSerTest(TransformType.COMPOSITE); }


	private void runTransformSerTest(TransformType type) {

		Map<String, Integer> spec = new HashMap<>();

		for(int k = 1; k < 5; k++) {
			for(int i = 1; i < 22; i++) {
				String cols = "";
				int j;
				for(j = 1; j <= i+1; j++){
					cols = cols + String.valueOf((int)((Math.random() * 22) + 1)) + " ";
				}
				cols = " " + (cols.substring(0, cols.length() - 1).replace(" ", ", ")) + " ";
				spec.put("{\n \"ids\": true\n, \"recode\":[" + cols + "]\n\n}", j-1);
				spec.put("{\n \"ids\": true\n, \"dummycode\":[" + cols + "]\n\n}", j-1);
			}
		}

		spec.put("{\n \"ids\": true\n, \"recode\":[ 2 ]\n\n}", 1);
		spec.put("{\n \"ids\": true\n, \"recode\":[ 2, 7 ]\n\n}", 2);
		spec.put("{\n \"ids\": true\n, \"recode\":[ 2, 7, 1 ]\n\n}", 3);
		spec.put("{\n \"ids\": true\n, \"recode\":[ 2, 7, 1, 4 ]\n\n}", 4);
		spec.put("{\n \"ids\": true\n, \"dummycode\":[ 2, 7, 1 ]\n\n}", 3);
		spec.put("{\n \"ids\": true\n, \"dummycode\":[ 2 ]\n\n}", 1);
		spec.put("{\n \"ids\": true\n, \"dummycode\":[ 2, 7 ]\n\n}", 2);

		switch( type ) {
			case RECODE: runRecode(); break;
			case DUMMY: runDummy(); break;
			case COMPOSITE:
				BufferedWriter writer = null;
				try {
					writer = new BufferedWriter(new FileWriter("measurements_recode"));
					for(String key : spec.keySet())
						runComposite(writer, key, spec.get(key));
					writer.close();
					break;
				}
				catch(IOException e) {
					e.printStackTrace();
				}

//			case BIN: runBin(); break;
			default: throw new RuntimeException("Unsupported transform type for encode/decode test.");
		}
	}
	
//	private void runTransformSerTest(String ofmt, TransformType type, boolean colnames)
//	{
//		//set transform specification
//		String SPEC = null;
//		switch( type ) {
//			case RECODE: SPEC = colnames?SPEC1b:SPEC1; break;
//			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; break;
//			default: throw new RuntimeException("Unsupported transform type for encode/decode test.");
//		}
//
//		String HOME = SCRIPT_DIR + TEST_DIR;
//
//		try {
//			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
//				new FileFormatPropertiesCSV(true, ",", false));
//			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);
//			String spec = "{\n \"ids\": true\n, \"recode\":[ 2 ]\n\n}";
//			String[] cnames = fb1.getColumnNames();
//
//			JSONObject jSpec = new JSONObject(spec);
//			EncoderRecode encoderIn = new EncoderRecode(jSpec, cnames, fb1.getNumColumns(), -1, -1);
//
//			FileOutputStream fileOutputStream = new FileOutputStream("myfile.txt");
//			ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
//			objectOutputStream.writeObject(encoderIn);
//			objectOutputStream.flush();
//			objectOutputStream.close();
//
//			FileInputStream fileInputStream
//				= new FileInputStream("myfile.txt");
//			ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
//			EncoderRecode encoderOut = (EncoderRecode) objectInputStream.readObject();
//			objectInputStream.close();
//
//			Assert.assertEquals(encoderIn.getCPRecodeMaps(), encoderOut.getCPRecodeMaps());
//
//		}
//		catch(IOException | JSONException | ClassNotFoundException e) {
//			e.printStackTrace();
//		}
//	}

	private void runComposite(BufferedWriter writer, String spec, int cols)
	{
		try {
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fin = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);

			Types.ValueType[] schema = UtilFunctions.nCopies(fin.getNumColumns(), Types.ValueType.STRING);

			fin.setSchema(schema);

			String[] cnames = fin.getColumnNames();

			long startTime = System.nanoTime();

			Encoder encoderIn = EncoderFactory.createEncoder(spec, cnames, fin.getNumColumns(), null);
			MatrixBlock data = encoderIn.encode(fin, new MatrixBlock(fin.getNumRows(), fin.getNumColumns(), false)); //build and apply
			FrameBlock meta = encoderIn.getMetaData(new FrameBlock(fin.getNumColumns(), Types.ValueType.STRING));
			meta.setColumnNames(cnames);

			EncoderComposite encoderOut = (EncoderComposite) writeReadCompare(encoderIn);

			long stopTime = System.nanoTime();
			writeToFile( writer,stopTime - startTime, cols);

			List<Encoder> eListIn = ((EncoderComposite) encoderIn).getEncoders();
			List<Encoder> eListOut = encoderOut.getEncoders();
			for(int i = 0; i < eListIn.size();  i++) {
				if(! (eListIn.get(i) instanceof EncoderPassThrough))
					Assert.assertEquals(eListIn.get(i), eListOut.get(i));
			}

		}
		catch(IOException  | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void writeToFile(BufferedWriter writer, long time, int cols)
		throws IOException {
//		System.out.println(time);
//		System.out.println(cols);
		String str = String.valueOf(cols) + " " + String.valueOf(time) + "\n";

		writer.write(str);
	}

	private void runRecode()
	{
		try {
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);
			String spec = "{\n \"ids\": true\n, \"recode\":[ 2 ]\n\n}";
			String[] cnames = fb1.getColumnNames();

			JSONObject jSpec = new JSONObject(spec);
			EncoderRecode encoderIn = new EncoderRecode(jSpec, cnames, fb1.getNumColumns(), -1, -1);
			encoderIn.encode(fb1, new MatrixBlock(fb1.getNumRows(), fb1.getNumColumns(), false));

			EncoderRecode encoderOut = (EncoderRecode) writeReadCompare(encoderIn);
			Assert.assertEquals(encoderIn.getCPRecodeMaps(), encoderOut.getCPRecodeMaps());
		}
		catch(IOException | JSONException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	private void runDummy()
	{
		try {
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);

			String spec = "{\n \"ids\": true\n, \"dummycode\":[ 2, 7, 1 ]\n\n}";
			String[] cnames = fb1.getColumnNames();

			JSONObject jSpec = new JSONObject(spec);
			EncoderDummycode encoderIn = new EncoderDummycode(jSpec, cnames, fb1.getNumColumns(), -1, -1);
			double[][] data = getNonZeroRandomMatrix(fb1.getNumRows(), fb1.getNumColumns(), 1, 1, 7);
			MatrixBlock out = new MatrixBlock(fb1.getNumRows(), fb1.getNumColumns(), false);
			out.init(data, fb1.getNumRows(), fb1.getNumColumns());
			encoderIn.initMetaData(fb1);
			encoderIn.encode(fb1, out);

			EncoderDummycode encoderOut = (EncoderDummycode) writeReadCompare(encoderIn);
			Assert.assertArrayEquals(encoderIn._domainSizes, encoderOut._domainSizes);
		}
		catch(IOException | JSONException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	private void runBin()
	{
		try {
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);
			// FIXME spec for bin
			String spec = "{\n \"ids\": true\n, \"bin\":[ 2 ]\n\n}";
			String[] cnames = fb1.getColumnNames();

			JSONObject jSpec = new JSONObject(spec);
			EncoderBin encoderIn = new EncoderBin(jSpec, cnames, fb1.getNumColumns(), -1, -1);
			MatrixBlock out = new MatrixBlock(fb1.getNumRows(), fb1.getNumColumns(), false);

			encoderIn.encode(fb1, out);

			EncoderBin encoderOut = (EncoderBin) writeReadCompare(encoderIn);
			Assert.assertEquals(encoderIn, encoderOut);
		}
		catch(IOException | JSONException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	private Encoder writeReadCompare (Encoder encoderIn) throws IOException, ClassNotFoundException
	{
		FileOutputStream fileOutputStream = new FileOutputStream(OUTPUT_FILE);
		ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
		objectOutputStream.writeObject(encoderIn);
		objectOutputStream.flush();
		objectOutputStream.close();

		FileInputStream fileInputStream = new FileInputStream(OUTPUT_FILE);
		ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
		Encoder encoderOut = (Encoder) objectInputStream.readObject();
		objectInputStream.close();

		Assert.assertArrayEquals(encoderIn.getColList(), encoderOut.getColList());
		Assert.assertEquals(encoderIn.getNumCols(), encoderOut.getNumCols());

		return encoderOut;
	}
}