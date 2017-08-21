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

package org.apache.sysml.test.integration.mlcontext.algorithms;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

import org.apache.log4j.Logger;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.test.integration.mlcontext.MLContextTestBase;
import org.junit.Assert;
import org.junit.Test;

public class MLContextSVMTest extends MLContextTestBase {
	protected static Logger log = Logger.getLogger(MLContextSVMTest.class);

	protected final static String TEST_SCRIPT_L2 = "scripts/algorithms/l2-svm.dml";
	protected final static String TEST_SCRIPT_M = "scripts/algorithms/m-svm.dml";
	
	private final static double eps = 1e-5;
	
	private final static int rows = 1468;
	private final static int cols = 987;
	
    private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int intercept = 0;
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	public enum SVMType {
	        L2,
	        M,
	}
  
  @Test
  public void testL2SVMSparse() {
          runSVMTestMLC(SVMType.L2, true);
  }
  
  public void testL2SVMDense() {
          runSVMTestMLC(SVMType.L2, false);
  }
  
  @Test
  public void testMSVMSparse() {
          runSVMTestMLC(SVMType.M, true);
  }
  
  @Test
  public void testMSVMDense() {
          runSVMTestMLC(SVMType.M, false);
  }
  
  
  private void runSVMTestMLC(SVMType type, boolean sparse) {
  
        // double[][] X = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 714);  
           double[][] X = getRandomMatrix(10, 3, 0, 1, sparse?sparsity2:sparsity1, 7);
           double[][] Y = getRandomMatrix(10, 1, 0, 10, 1.0, 3);
           
           switch(type) {
               case L2:
                       Script l2svm = dmlFromFile(TEST_SCRIPT_L2);
                       l2svm.in("X", X).in("Y", Y).in("$icpt", intercept).in("$tol", epsilon).in("$reg", "0.001").in("$maxiter", maxiter).out("w");   
                       ml.execute(l2svm);
                       
                       break;
                       
               case M:
                       Script msvm = dmlFromFile(TEST_SCRIPT_M);
                       msvm.in("X", X).in("Y", Y).in("$icpt", intercept).in("$tol", epsilon).in("$reg", "0.001").in("$maxiter", maxiter).out("w");
                       ml.execute(msvm);
                       
                       break;
           }
  }
}
