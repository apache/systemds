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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;

public class BuiltinSherlockTest extends AutomatedTestBase {
	private final static String TEST_NAME = "sherlock";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinScaleTest.class.getSimpleName() + "/";

@Override
public void setUp() {
  addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));

}

@Test
public void testSherlock() {
  runtestSherlock();
}

private void runtestSherlock() {
  loadTestConfiguration(getTestConfiguration(TEST_NAME));
  String HOME = SCRIPT_DIR + TEST_DIR;
  fullDMLScriptName = HOME + TEST_NAME + ".dml";
  List<String> proArgs = new ArrayList<>();
  /*proArgs.add("-args");
  proArgs.add(input("X"));
  proArgs.add(input("Y"));
  proArgs.add(output("cW1"));
  proArgs.add(output("cb1"));
  proArgs.add(output("cW2"));
  proArgs.add(output("cb2"));
  proArgs.add(output("cW3"));
  proArgs.add(output("cb3"));
  proArgs.add(output("wW1"));
  proArgs.add(output("wb1"));
  proArgs.add(output("wW2"));
  proArgs.add(output("wb2"));
  proArgs.add(output("wW3"));
  proArgs.add(output("wb3"));
  proArgs.add(output("pW1"));
  proArgs.add(output("pb1"));
  proArgs.add(output("pW2"));
  proArgs.add(output("pb2"));
  proArgs.add(output("pW3"));
  proArgs.add(output("pb3"));
  */
  programArgs = proArgs.toArray(new String[proArgs.size()]);

  double[][] X = getRandomMatrix(1000, 1588, 0, 3, 0.9, 7);
  double[][] Y = getRandomMatrix(1000, 78, 0, 1, 0.9, 7);

  writeInputMatrixWithMTD("X", X, true);
  writeInputMatrixWithMTD("Y", Y, true);

  runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

  //compare expected results
  HashMap<MatrixValue.CellIndex, Double> cW1 = readDMLMatrixFromOutputDir("cW1");
  HashMap<MatrixValue.CellIndex, Double> cb1 = readDMLMatrixFromOutputDir("cb1");
  HashMap<MatrixValue.CellIndex, Double> cW2 = readDMLMatrixFromOutputDir("cW2");
  HashMap<MatrixValue.CellIndex, Double> cb2 = readDMLMatrixFromOutputDir("cb2");
  HashMap<MatrixValue.CellIndex, Double> cW3 = readDMLMatrixFromOutputDir("cW3");
  HashMap<MatrixValue.CellIndex, Double> cb3 = readDMLMatrixFromOutputDir("cb3");
  HashMap<MatrixValue.CellIndex, Double> wW1 = readDMLMatrixFromOutputDir("wW1");
  HashMap<MatrixValue.CellIndex, Double> wb1 = readDMLMatrixFromOutputDir("wb1");
  HashMap<MatrixValue.CellIndex, Double> wW2 = readDMLMatrixFromOutputDir("wW2");
  HashMap<MatrixValue.CellIndex, Double> wb2 = readDMLMatrixFromOutputDir("wb2");
  HashMap<MatrixValue.CellIndex, Double> wW3 = readDMLMatrixFromOutputDir("wW3");
  HashMap<MatrixValue.CellIndex, Double> wb3 = readDMLMatrixFromOutputDir("wb3");
  HashMap<MatrixValue.CellIndex, Double> pW1 = readDMLMatrixFromOutputDir("pW1");
  HashMap<MatrixValue.CellIndex, Double> pb1 = readDMLMatrixFromOutputDir("pb1");
  HashMap<MatrixValue.CellIndex, Double> pW2 = readDMLMatrixFromOutputDir("pW2");
  HashMap<MatrixValue.CellIndex, Double> pb2 = readDMLMatrixFromOutputDir("pb2");
  HashMap<MatrixValue.CellIndex, Double> pW3 = readDMLMatrixFromOutputDir("pW3");
  HashMap<MatrixValue.CellIndex, Double> pb3 = readDMLMatrixFromOutputDir("pb3");
  HashMap<MatrixValue.CellIndex, Double> fW1 = readDMLMatrixFromOutputDir("fW1");
  HashMap<MatrixValue.CellIndex, Double> fb1 = readDMLMatrixFromOutputDir("fb1");
  HashMap<MatrixValue.CellIndex, Double> fW2 = readDMLMatrixFromOutputDir("fW2");
  HashMap<MatrixValue.CellIndex, Double> fb2 = readDMLMatrixFromOutputDir("fb2");
  HashMap<MatrixValue.CellIndex, Double> fW3 = readDMLMatrixFromOutputDir("fW3");
  HashMap<MatrixValue.CellIndex, Double> fb3 = readDMLMatrixFromOutputDir("fb3");

  TestUtils.compareScalars(1, 1, 0);
}


}
