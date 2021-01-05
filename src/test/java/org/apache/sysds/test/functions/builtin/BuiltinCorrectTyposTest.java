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

package org.apache.sysds.test.functions.builtin;

// import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
// import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.IOException;

public class BuiltinCorrectTyposTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "correct_typos";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinCorrectTyposTest.class.getSimpleName() + "/";
	
	private final static Types.ValueType[] schema = {Types.ValueType.STRING};

	// private final static double eps = 1e-10;
	// private final static int rows = 70;
	// private final static int cols = 50;
	// private final static double spSparse = 0.1;
	// private final static double spDense = 0.9;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

  @Test
  public void testCorrectTyposCP() throws IOException {
    runCorrectTyposTest(true, ExecType.CP);
  }

  // TODO: this test fails unless the new frames are printed before accessing them
  // @Test
  // public void testCorrectTyposSP() throws IOException {
    // runCorrectTyposTest(true, ExecType.SPARK);
  // }

	
	private void runCorrectTyposTest(boolean decapitalize, ExecType instType) throws IOException
	{
		ExecMode platformOld = setExecMode(instType);

    System.out.println("Begin CorrectTyposTest");
		
    try
    {
      loadTestConfiguration(getTestConfiguration(TEST_NAME));

      String HOME = SCRIPT_DIR + TEST_DIR;
      fullDMLScriptName = HOME + TEST_NAME + ".dml";

      fullRScriptName = HOME + TEST_NAME + ".R";
      programArgs = new String[]{
        "-nvargs", "X=" + input("X"), "Y=" + output("Y"),
        "decapitalize=" + decapitalize};
      rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

      System.out.println("Create dataset");
      FrameBlock frame = new FrameBlock(schema);
      FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV);
      int rows = initFrameData(frame);
      int cols = 1;
      // System.out.println("Write dataset");
      // System.out.println(frame.getNumColumns());
			writer.writeFrameToHDFS(frame.slice(0, rows - 1, 0, 0, new FrameBlock()), input("X"), rows, cols);

      System.out.println("Run test");
      runTest(true, false, null, -1);
      System.out.println("DONE");

      //compare matrices
      // HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
      // HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
      // TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
    }
    finally {
      rtplatform = platformOld;
    }
	}

  private static int initFrameData(FrameBlock frame) {
    String[] strings = new String[] {"AuStRiA", "Austria", "AUSTRIA", "India", "IIT", "INDIA", "India", "Pakistan", "PakistaN", "Austria", "Austria"};
    frame.appendColumn(strings);

    return strings.length;
  }
}
