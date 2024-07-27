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
package org.apache.sysds.test.functions.dnn;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;
import java.util.Objects;

public class LSTMTest extends AutomatedTestBase {
	String TEST_NAME1 = "LSTMForwardTest";
	String TEST_NAME2 = "LSTMBackwardTest";
	String TEST_NAME3 = "BILSTMForwardTest";
	String TEST_NAME4 = "BILSTMBackwardTest";
	private final static String TEST_DIR = "functions/tensor/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4));
	}

	@Test
	public void testLSTMForwardLocalSingleSample1(){
		runLSTMTest(1, 32, 1,1, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocalSingleSample2(){
		runLSTMTest(1, 1, 64,1, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocalSingleSample3(){
		runLSTMTest(1, 1, 1,2048, TEST_NAME1);
	}

	//note elias: for large hidden sizes there is discrepancy between built-in and the dml script
	@Test
	public void testLSTMForwardLocalSingleSample4(){
		runLSTMTest(1, 32, 32,1025, 0,0, 1e-2, TEST_NAME1,false);
	}

	@Test
	public void testLSTMForwardLocal1(){
		runLSTMTest(64, 2, 2,2, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocal2(){
		runLSTMTest(32, 8, 1,1, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocal3(){
		runLSTMTest(32, 1, 64,1, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocal4(){
		runLSTMTest(32, 8, 36,1025, TEST_NAME1);
	}

	@Test
	public void testLSTMForwardLocal5(){
		runLSTMTest(32, 75, 128,256, 0, 1, 1e-3, TEST_NAME1, false);
	}

	@Test
	public void testLSTMBackwardLocalSingleSample1(){
		runLSTMTest(1, 2, 3,4,0,1,1e-5, TEST_NAME2, true);
	}

	@Test
	public void testLSTMBackwardLocal1(){
		runLSTMTest(64, 32, 16,32,0,0,1e-5, TEST_NAME2, true);
	}

	@Test
	public void testLSTMBackwardLocal2(){
		runLSTMTest(64, 32, 16,32,0,1,1e-5, TEST_NAME2, true);
	}

	@Test
	@Ignore
	public void testLSTMForwardLocalLarge(){
		runLSTMTest(100, 32, 128,64, 0, 1, 1e-5, TEST_NAME1, false);
	}

	@Test
	@Ignore
	public void testLSTMBackwardLocalLarge(){
		runLSTMTest(128, 128, 128,64, 0, 0, 1e-5, TEST_NAME2, true);
	}

	// The BILSTM output is compared to output of pytorch's BI-LSTM Layer implementation with FP64.
	// Expected results are saved at: "src/test/resources/expected/BILSTM_OUT_{batch_size}_{seq_length}_{num_features}_{hidden_size}.csv"
	@Test
	public void testBILSTMForwardLocal1(){
		runLSTMTest(3, 5, 2,2, 0, 1, 1e-5, TEST_NAME3,false);
	}

	@Test
	public void testBILSTMForwardLocal2(){
		runLSTMTest(6, 5, 6,4, 0, 1, 1e-5, TEST_NAME3,false);
	}

	@Test
	public void testBILSTMForwardLocal3(){
		runLSTMTest(10, 5, 2,6, 0, 1, 1e-5, TEST_NAME3,false);
	}

	@Test
	public void testBILSTMForwardLocal4(){
		runLSTMTest(3, 5, 2,2, 0, 0, 1e-5, TEST_NAME3,false);
	}

	@Test
	public void testBILSTMForwardLocal5(){
		runLSTMTest(2, 5, 2,2, 0, 1, 1e-5, TEST_NAME3,false);
	}


	@Test
	public void testBILSTMBackwardLocal1(){
		runLSTMTest(10, 5, 2,6, 0, 1, 1e-5, TEST_NAME4,false);
	}

	@Test
	public void testBILSTMFBackwardLocal2(){
		runLSTMTest(1, 5, 6,4, 0, 1, 1e-5, TEST_NAME4,false);
	}

	@Test
	public void testBILSTMBackwardLocal3(){
		runLSTMTest(6, 5, 6,4, 0, 1, 1e-5, TEST_NAME4,false);
	}

	@Test
	public void testBILSTMBackwardLocal4(){
		runLSTMTest(5, 5, 6,4, 0, 1, 1e-5, TEST_NAME4,false);
	}

	@Test
	public void testBILSTMBackwardLocal5(){
		runLSTMTest(4, 5, 6,4, 0, 1, 1e-5, TEST_NAME4,false);
	}


	private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, String testname){
		runLSTMTest(batch_size, seq_length, num_features, hidden_size,0, testname);
	}

	private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, int debug, String testname){
		runLSTMTest(batch_size, seq_length, num_features, hidden_size,debug, 0, 1e-5, testname, false);
	}

	private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, int debug, int seq,  double precision, String testname, boolean backward)
	{
		//set runtime platform
		Types.ExecMode rtold = setExecMode(Types.ExecMode.SINGLE_NODE);
		try
		{
			getAndLoadTestConfiguration(testname);
			fullDMLScriptName = getScript();

			//run script
			//"-explain", "runtime",
			boolean bilstm = Objects.equals(testname, TEST_NAME3);
			boolean bilstm_backwards = Objects.equals(testname, TEST_NAME4);
			if(bilstm)
				programArgs = new String[]{"-stats","-args", String.valueOf(batch_size), String.valueOf(seq_length),
						String.valueOf(num_features), String.valueOf(hidden_size), String.valueOf(debug), String.valueOf(seq),
						"src/test/resources/expected/BILSTM_OUT",output("1A")};
			else if(bilstm_backwards)
				programArgs = new String[]{"-stats","-args", String.valueOf(batch_size), String.valueOf(seq_length),
						String.valueOf(num_features), String.valueOf(hidden_size), String.valueOf(debug), String.valueOf(seq),
						"src/test/resources/expected/BILSTM_back_dW",
						"src/test/resources/expected/BILSTM_back_dc",
						"src/test/resources/expected/BILSTM_back_dX", output("1A")};
			else{
				programArgs = new String[]{"-stats","-args", String.valueOf(batch_size), String.valueOf(seq_length),
						String.valueOf(num_features), String.valueOf(hidden_size), String.valueOf(debug), String.valueOf(seq),
						output("1A"),output("1B"),output("2A"), output("2B"),output("3A"),output("3B"),"","","",""};
				int offset = 0;
				if(backward){
					programArgs[14 + offset] = output("4A");
					programArgs[15 + offset] = output("4B");
					programArgs[16 + offset] = output("5A");
					programArgs[17 + offset] = output("5B");
				}
			}

			//output("4A"), output("4B"),output("5A"),output("5B")
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			// Compare results
			if(bilstm){
				Double max_error = (Double) readDMLScalarFromOutputDir("1A").values().toArray()[0];
				assert max_error < precision;
			} else if (bilstm_backwards) {
				HashMap<MatrixValue.CellIndex, Double> errors = readDMLMatrixFromOutputDir("1A");
				double[][] errors_ = TestUtils.convertHashMapToDoubleArray(errors);
				assert errors_[0][0] < precision;
				assert errors_[0][1] < precision;
				assert errors_[0][2] < precision;
			} else{
				extracted(precision,"1");
				extracted(precision,"2");
				extracted(precision,"3");
				if(backward){
					extracted(precision,"4");
					extracted(precision,"5");
				}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(rtold);
		}
	}

	private void extracted(double precision, String output) {
		HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir(output+"A");
		double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
		HashMap<MatrixValue.CellIndex, Double> res_expected = readDMLMatrixFromOutputDir(output+"B");
		double[][] resultExpectedDouble = TestUtils.convertHashMapToDoubleArray(res_expected);
		TestUtils.compareMatrices(resultExpectedDouble, resultActualDouble, precision);
	}
}
