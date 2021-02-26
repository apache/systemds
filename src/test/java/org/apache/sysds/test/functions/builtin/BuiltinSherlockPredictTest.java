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

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BuiltinSherlockPredictTest extends AutomatedTestBase {
  private final static String TEST_NAME = "sherlockPredict";
  private final static String TEST_DIR = "functions/builtin/";
  private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSherlockPredictTest.class.getSimpleName() + "/";
  private final static String WEIGHTS_DIR = SCRIPT_DIR + TEST_DIR + "data/sherlockWeights/";
  @Override public void setUp() {
    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
  }

  @Test public void testSherlockPredict() {
    runtestSherlockPredict();
  }

  private void runtestSherlockPredict() {
    loadTestConfiguration(getTestConfiguration(TEST_NAME));
    String HOME = SCRIPT_DIR + TEST_DIR;
    fullDMLScriptName = HOME + TEST_NAME + ".dml";

    List<String> proArgs = new ArrayList<>();
    proArgs.add("-exec");
    proArgs.add(" singlenode");
    proArgs.add("-nvargs");
    proArgs.add("X=" +   input("X"));
    proArgs.add("cW1=" + WEIGHTS_DIR + "cW1");
    proArgs.add("cb1=" + WEIGHTS_DIR + "cb1");
    proArgs.add("cW2=" + WEIGHTS_DIR + "cW2");
    proArgs.add("cb2=" + WEIGHTS_DIR + "cb2");
    proArgs.add("cW3=" + WEIGHTS_DIR + "cW3");
    proArgs.add("cb3=" + WEIGHTS_DIR + "cb3");
    proArgs.add("wW1=" + WEIGHTS_DIR + "wW1");
    proArgs.add("wb1=" + WEIGHTS_DIR + "wb1");
    proArgs.add("wW2=" + WEIGHTS_DIR + "wW2");
    proArgs.add("wb2=" + WEIGHTS_DIR + "wb2");
    proArgs.add("wW3=" + WEIGHTS_DIR + "wW3");
    proArgs.add("wb3=" + WEIGHTS_DIR + "wb3");
    proArgs.add("pW1=" + WEIGHTS_DIR + "pW1");
    proArgs.add("pb1=" + WEIGHTS_DIR + "pb1");
    proArgs.add("pW2=" + WEIGHTS_DIR + "pW2");
    proArgs.add("pb2=" + WEIGHTS_DIR + "pb2");
    proArgs.add("pW3=" + WEIGHTS_DIR + "pW3");
    proArgs.add("pb3=" + WEIGHTS_DIR + "pb3");
    proArgs.add("fW1=" + WEIGHTS_DIR + "fW1");
    proArgs.add("fb1=" + WEIGHTS_DIR + "fb1");
    proArgs.add("fW2=" + WEIGHTS_DIR + "fW2");
    proArgs.add("fb2=" + WEIGHTS_DIR + "fb2");
    proArgs.add("fW3=" + WEIGHTS_DIR + "fW3");
    proArgs.add("fb3=" + WEIGHTS_DIR + "fb3");
    proArgs.add("probs=" + output("probs"));
    programArgs = proArgs.toArray(new String[proArgs.size()]);

    double[][] X = getRandomMatrix(256, 1588, 0, 3, 0.9, 7);

    writeInputMatrixWithMTD("X", X, true);
    runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

    //read probs
    HashMap<MatrixValue.CellIndex, Double> probs = readDMLMatrixFromOutputDir("probs");
  }
}