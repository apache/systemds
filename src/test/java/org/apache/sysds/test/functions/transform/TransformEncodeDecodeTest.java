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

import java.util.HashMap;
import java.util.Iterator;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 */
public class TransformEncodeDecodeTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "TransformEncodeDecode";
	private static final String TEST_DIR = "functions/transform/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TransformEncodeDecodeTest.class.getSimpleName() + "/";
	
	private static final String SPEC = "TransformEncodeDecodeSpec.json";
	
	private static final int rows = 1234;
	private static final int cols = 2;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"FO"}));
	}
	
	@Test
	public void runTestCSVDenseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, false, "csv");
	}
	
	@Test
	public void runTestCSVSparseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, true, "csv");
	}
	
	@Test
	public void runTestTextcellDenseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, false, "text");
	}
	
	@Test
	public void runTestTextcellSparseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, true, "text");
	}
	
	@Test
	public void runTestBinaryDenseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, false, "binary");
	}
	
	@Test
	public void runTestBinarySparseCP() {
		runTransformEncodeDecodeTest(ExecType.CP, true, "binary");
	}
	
	private void runTransformEncodeDecodeTest( ExecType et, boolean sparse, String fmt)
	{
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID; //only CP supported

		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			//generate and write input data
			double[][] A = TestUtils.round(getRandomMatrix(rows, cols, 1, 15, sparse ? sparsity2 : sparsity1, 7)); 
			FrameBlock FA = DataConverter.convertToFrameBlock(DataConverter.convertToMatrixBlock(A));  
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.safeValueOf(fmt));
			writer.writeFrameToHDFS(FA, input("F"), rows, cols);
			
			fullDMLScriptName = SCRIPT_DIR+TEST_DIR+TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain","-args", input("F"), fmt, 
					String.valueOf(rows), String.valueOf(cols), 
					SCRIPT_DIR+TEST_DIR+SPEC, output("FO") };
			
			//run test
			runTest(true, false, null, -1); 
			
			//compare matrices (values recoded to identical codes)
			FrameReader reader = FrameReaderFactory.createFrameReader(FileFormat.safeValueOf(fmt));
			FrameBlock FO = reader.readFrameFromHDFS(output("FO"), 16, 2);
			HashMap<String,Long> cFA = getCounts(FA, 1);
			Iterator<String[]> iterFO = FO.getStringRowIterator();
			while( iterFO.hasNext() ) {
				String[] row = iterFO.next();
				Double expected = (double)cFA.get(row[1]);
				Double val = (row[0]!=null)?Double.valueOf(row[0]) : 0;
				Assert.assertEquals("Output aggregates don't match: "+
						expected+" vs "+val, expected, val);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail(ex.getMessage());
		}
		finally {
			rtplatform = platformOld;
		}
	}	
	
	private static HashMap<String, Long> getCounts(FrameBlock input, int groupBy) {
		HashMap<String, Long> ret = new HashMap<>();
		for( int i=0; i<input.getNumRows(); i++ ) {
			String key = input.get(i, groupBy).toString();
			Long tmp = ret.get(key);
			ret.put(key, (tmp!=null)?tmp+1:1);
		}
		return ret;
	}
}
