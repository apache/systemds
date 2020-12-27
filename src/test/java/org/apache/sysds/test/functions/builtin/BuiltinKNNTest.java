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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class BuiltinKNNTest extends AutomatedTestBase
{
  private final static Log LOG = LogFactory.getLog(BuiltinKNNTest.class.getName());

  private final static String TEST_NAME = "KNN";
  private final static String TEST_DIR = "functions/builtin/";
  private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinKNNTest.class.getSimpleName() + "/";

  private final static String OUTPUT_NAME = "B";

  private final static Double TEST_TOLERANCE = 1e-9;

  public int blocksize = 1024;
  @Parameterized.Parameter()
  public int rows;
  @Parameterized.Parameter(1)
  public int cols;
  @Parameterized.Parameter(2)
  public int query_rows;
  @Parameterized.Parameter(3)
  public int query_cols;
  @Parameterized.Parameter(4)
  public boolean continuous;
  @Parameterized.Parameter(5)
  public String fmt;
  @Parameterized.Parameter(6)
  public double sparsity;

  @Override
  public void setUp()
  {
    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, new String[] {OUTPUT_NAME}));
  }

  @Parameterized.Parameters
  public static Collection<Object[]> data()
  {
    return Arrays.asList(new Object[][] {
      // {rows, cols, query_rows, query_cols, continuous, fmt, sparsity}
      {1000, 500, 35, 450, true, "csv", 0.1},
      {1000, 500, 35, 450, true, "csv", 0.9}
    });
  }

  @Test
  public void testKNN()
  {
    runKNNTest(ExecMode.SINGLE_NODE);
  }

  private void runKNNTest(ExecMode exec_mode)
  {
    ExecMode platform_old = setExecMode(exec_mode);

    getAndLoadTestConfiguration(TEST_NAME);

    String HOME = SCRIPT_DIR + TEST_DIR;

    double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 255);
    double[][] T = getRandomMatrix(query_rows, query_cols, 0, 1, 1, 65);

    writeInputMatrixWithMTD("X", X, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
    writeInputMatrixWithMTD("T", T, false, new MatrixCharacteristics(query_rows, query_cols, blocksize, rows * cols));

    fullDMLScriptName = HOME + TEST_NAME + ".dml";
    programArgs = new String[] {"-exec", "-args",
      "in_X=" + input("X"), "in_T=" + input("T"), "in_continuous=" + (continuous ? "1" : "0"), "in_format=" + fmt,
      "out_B=" + output(OUTPUT_NAME)};

    fullRScriptName = HOME + TEST_NAME + ".R";
    rCmd = "Rscript" + " " + fullRScriptName + " " +
      input("X") + " " + input("T") + " " + (continuous ? "1" : "0") + " " + fmt + " " +
      expectedDir();

    runTest(true, false, null, -1);
    runRScript(true);

    compareResultsWithR(TEST_TOLERANCE);

    // restore execution mode
    setExecMode(platform_old);
  }
}
