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

import org.apache.sysds.common.Types.ExecMode;
// import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.lang.Math;
import java.util.Random;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class BuiltinMatrixProfileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "matrix_profile";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMatrixProfileTest.class.getSimpleName() + "/";
	
  private static Random generator;
  private final static int seed = 42;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	 @Test
	 public void testMatrixProfileCP() throws IOException {
		 runMatrixProfileTest(4, "TRUE", ExecType.CP);
	 }

   // @Test
	 // public void testMatrixProfileSPARK() throws IOException {
		 // runMatrixProfileTest(4, "TRUE", ExecType.SPARK);
	 // }

	
	private void runMatrixProfileTest(Integer window_size, String is_verbose, ExecType instType) throws IOException
	{
		ExecMode platformOld = setExecMode(instType);

		System.out.println("Begin MatrixProfileTest");
    
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
				"-nvargs", "TS=" + input("TS"), "MP=" + output("MP"),
        "MPI=" + output("MPI"),
				"window_size=" + window_size,
				"is_verbose=" + is_verbose};

			generator = new Random(seed);

      int len = 50;
      // double[][] ts = initTimeSeries(10);
      double[][] ts = genSineWave(len, 0.05, 1, 1);
      int[] noise_idxs = addNoise(1, len, ts);
      writeInputMatrixWithMTD("TS", ts, false);

			System.out.println("Run test");
			runTest(true, false, null, -1);
			System.out.println("DONE");

			HashMap<CellIndex, Double> MP = readDMLMatrixFromOutputDir("MP");
			HashMap<CellIndex, Double> MPI = readDMLMatrixFromOutputDir("MPI");

      List sortedList = sortByValues(MP);
			// System.out.println(sortedList);
      Map.Entry entry = (Map.Entry)sortedList.get(0);
      System.out.println(entry);
      int highest_dist_idx = ((CellIndex)entry.getKey()).row;
      System.out.println(highest_dist_idx);

      int noise_idx = noise_idxs[0];
      Assert.assertTrue(highest_dist_idx>noise_idx-window_size || highest_dist_idx<noise_idx+window_size);
		}
		finally {
			rtplatform = platformOld;
		}
	}

  private static double[][] initTimeSeries(Integer n) {
    double[][] ts = new double[n][1];
    for (int i=0; i<n; ++i) {
      ts[i][0] = i;
    }
    return ts;
  }

  private static double[][] genSineWave(Integer n, double sampling_rate, float p, float amp) {
    double[][] ts = new double[n][1];
    for (int i=0; i<n; ++i) {
      ts[i][0] = p*Math.sin(amp*sampling_rate*i);
    }
    return ts;
  }
  
  private static int[] addNoise(Integer n, Integer len, double[][] ts) {
    int[] idxs = new int[n];
    for (int i=0; i<n; ++i) {
      int idx = generator.nextInt(len);
      System.out.println("Add noise to idx: " + idx);
      ts[idx][0] += 1;
      idxs[i] = idx;
    }
    return idxs;
  }

  private static boolean verifyResult(double[][] matrix_profile, 
      int[][] matrix_profile_index, int[] perturbation_idxs) {

    return true;
  }

  private static List sortByValues(HashMap map) {
    List list = new LinkedList(map.entrySet());
    Collections.sort(list, new Comparator() {
      public int compare(Object o1, Object o2) {
        return ((Comparable) ((Map.Entry) (o2)).getValue())
          .compareTo(((Map.Entry) (o1)).getValue());
      }
    });
    return list;
  }
}
