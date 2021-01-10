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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

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
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderRecode;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.Statistics;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;
import org.junit.Assert;
import org.junit.Test;

public class FrameEncodeDecodeSerTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "TransformFrameEncodeDecode";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameEncodeDecodeSerTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 	= "homes3/homes.csv";
	private final static String SPEC1 		= "homes3/homes.tfspec_recode.json"; 
	private final static String SPEC1b 		= "homes3/homes.tfspec_recode2.json"; 
	private final static String SPEC2 		= "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b 		= "homes3/homes.tfspec_dummy2.json";
	
	public enum TransformType {
		RECODE,
		DUMMY,
		BIN,
		IMPUTE,
		OMIT,
	}

	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
	}

	@Test
	public void testDummy() {
		runTransformSerTest("csv", TransformType.DUMMY, false);
	}

	@Test
	public void testRecode() {
		runTransformSerTest("csv", TransformType.RECODE, false);
	}
	
	private void runTransformSerTest(String ofmt, TransformType type, boolean colnames)
	{
		//set transform specification
		String SPEC = null; String DATASET = null;
		switch( type ) {
			case RECODE: SPEC = colnames?SPEC1b:SPEC1; DATASET = DATASET1; break;
			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; DATASET = DATASET1; break;
			default: throw new RuntimeException("Unsupported transform type for encode/decode test.");
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");

		String HOME = SCRIPT_DIR + TEST_DIR;

		try {
			FrameReader reader1 = FrameReaderFactory.createFrameReader(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock fb1 = reader1.readFrameFromHDFS(HOME + "input/" + DATASET, -1L, -1L);
			String spec = "{\n \"ids\": true\n, \"recode\":[ 2 ]\n\n}";
			String[] cnames = fb1.getColumnNames();

//			Encoder encoder = EncoderFactory.createEncoder(spec, cnames, fb1.getNumColumns(), null);
			JSONObject jSpec = new JSONObject(spec);
			EncoderRecode encoderIn = new EncoderRecode(jSpec, cnames, fb1.getNumColumns(), -1, -1);

			FileOutputStream fileOutputStream = new FileOutputStream("myfile.txt");
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(encoderIn);
			objectOutputStream.flush();
			objectOutputStream.close();

			FileInputStream fileInputStream
				= new FileInputStream("myfile.txt");
			ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
			EncoderRecode encoderOut = (EncoderRecode) objectInputStream.readObject();
			objectInputStream.close();

			Assert.assertEquals(encoderIn.getCPRecodeMaps(), encoderOut.getCPRecodeMaps());

		}
		catch(IOException | JSONException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
}