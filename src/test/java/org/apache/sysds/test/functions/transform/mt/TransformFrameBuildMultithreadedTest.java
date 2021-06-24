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

package org.apache.sysds.test.functions.transform.mt;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.Assert.*;

public class TransformFrameBuildMultithreadedTest  extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameBuildMultithreadedTest";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameBuildMultithreadedTest.class.getSimpleName() + "/";

	// dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1 = "homes3/homes.tfspec_recode.json";
	private final static String SPEC1b = "homes3/homes.tfspec_recode2.json";
	private final static String SPEC2 = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b = "homes3/homes.tfspec_dummy2.json";
	private final static String SPEC3 = "homes3/homes.tfspec_bin.json"; // recode
	private final static String SPEC3b = "homes3/homes.tfspec_bin2.json"; // recode
	private final static String SPEC6 = "homes3/homes.tfspec_recode_dummy.json";
	private final static String SPEC6b = "homes3/homes.tfspec_recode_dummy2.json";
	private final static String SPEC7 = "homes3/homes.tfspec_binDummy.json"; // recode+dummy
	private final static String SPEC7b = "homes3/homes.tfspec_binDummy2.json"; // recode+dummy
	private final static String SPEC8 = "homes3/homes.tfspec_hash.json";
	private final static String SPEC8b = "homes3/homes.tfspec_hash2.json";
	private final static String SPEC9 = "homes3/homes.tfspec_hash_recode.json";
	private final static String SPEC9b = "homes3/homes.tfspec_hash_recode2.json";
	private final static String SPEC10 = "homes3/homes.tfspec_recode_bin.json";

	public enum TransformType {
		RECODE, DUMMY, RECODE_DUMMY, BIN, BIN_DUMMY, HASH, HASH_RECODE, RECODE_BIN,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"y"}));
	}

	@Test
	public void testHomesRecodeIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, false);
	}

	@Test
	public void testHomesDummyCodeIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}

	@Test
	public void testHomesRecodeDummyCodeIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, false);
	}

	@Test
	public void testHomesRecodeBinningIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_BIN, false);
	}

	@Test
	public void testHomesBinIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.BIN, false);
	}

	@Test
	public void testHomesBinDummyIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, false);
	}

	@Test
	public void testHomesHashIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.HASH, false);
	}

	@Test
	public void testHomesHashRecodeIDsSingleNodeCSV() {
		runTransformTest(Types.ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, false);
	}


	private void runTransformTest(Types.ExecMode rt, String ofmt, TransformType type, boolean colnames) 
	{
		// set transform specification
		String SPEC = null;
		String DATASET = null;
		switch (type) {
			case RECODE:
				SPEC = colnames ? SPEC1b : SPEC1;
				DATASET = DATASET1;
				break;
			case DUMMY:
				SPEC = colnames ? SPEC2b : SPEC2;
				DATASET = DATASET1;
				break;
			case BIN:
				SPEC = colnames ? SPEC3b : SPEC3;
				DATASET = DATASET1;
				break;
			case RECODE_DUMMY:
				SPEC = colnames ? SPEC6b : SPEC6;
				DATASET = DATASET1;
				break;
			case BIN_DUMMY:
				SPEC = colnames ? SPEC7b : SPEC7;
				DATASET = DATASET1;
				break;
			case HASH:
				SPEC = colnames ? SPEC8b : SPEC8;
				DATASET = DATASET1;
				break;
			case HASH_RECODE:
				SPEC = colnames ? SPEC9b : SPEC9;
				DATASET = DATASET1;
				break;
			case RECODE_BIN:
				SPEC = colnames ? SPEC10 : SPEC10;
				DATASET = DATASET1;
				break;
		}

		if (!ofmt.equals("csv"))
			throw new RuntimeException("Unsupported test output format");

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			//String HOME = SCRIPT_DIR + TEST_DIR;
			DATASET = DATASET_DIR + DATASET;
			SPEC = DATASET_DIR + SPEC;

			FileFormatPropertiesCSV props = new FileFormatPropertiesCSV();
			props.setHeader(true);
			FrameBlock input = FrameReaderFactory.createFrameReader(Types.FileFormat.CSV, props)
					.readFrameFromHDFS(DATASET, -1L, -1L);
			StringBuilder specSb = new StringBuilder();
			Files.readAllLines(Paths.get(SPEC)).forEach(s -> specSb.append(s).append("\n"));
			MultiColumnEncoder encoderS = EncoderFactory.createEncoder(specSb.toString(), 
					input.getColumnNames(), input.getNumColumns(), null);
			MultiColumnEncoder encoderM = EncoderFactory.createEncoder(specSb.toString(), 
					input.getColumnNames(), input.getNumColumns(), null);

			MultiColumnEncoder.BUILD_BLOCKSIZE = 10;
			encoderS.build(input, 1);
			encoderM.build(input, 12);

			if (type == TransformType.RECODE) {
				List<ColumnEncoderRecode> encodersS = encoderS.getColumnEncoders(ColumnEncoderRecode.class);
				List<ColumnEncoderRecode> encodersM = encoderM.getColumnEncoders(ColumnEncoderRecode.class);
				assertEquals(encodersS.size(), encodersM.size());
				for (int i = 0; i < encodersS.size(); i++) {
					assertEquals(encodersS.get(i).getRcdMap().keySet(), encodersM.get(i).getRcdMap().keySet());
				}
			}
			else if (type == TransformType.BIN) {
				List<ColumnEncoderBin> encodersS = encoderS.getColumnEncoders(ColumnEncoderBin.class);
				List<ColumnEncoderBin> encodersM = encoderM.getColumnEncoders(ColumnEncoderBin.class);
				assertEquals(encodersS.size(), encodersM.size());
				for (int i = 0; i < encodersS.size(); i++) {
					assertArrayEquals(encodersS.get(i).getBinMins(), encodersM.get(i).getBinMins(), 0);
					assertArrayEquals(encodersS.get(i).getBinMaxs(), encodersM.get(i).getBinMaxs(), 0);
				}
			}
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
	}

}
