/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.builtin;


import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class BuiltinSliceFinderTest extends AutomatedTestBase {

	private final static String TEST_NAME = "slicefinder";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSliceFinderTest.class.getSimpleName() + "/";

	private final static int rows = 2000;
	private final static int cols = 10;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
	}

	@Test
	public void SingleFreatureTest() {
		runslicefindertest(1,true, ExecType.CP, BuiltinLmTest.LinregType.AUTO);
	}

	@Test
	public void MultipleValuesOneFeature() {
		runslicefindertest(2,true, ExecType.CP, BuiltinLmTest.LinregType.AUTO);
	}

	@Test
	public void MultipleFeaturesSingleValues() {
		runslicefindertest(3,true, ExecType.CP, BuiltinLmTest.LinregType.AUTO);
	}

	private void runslicefindertest(int test,boolean sparse, ExecType instType, BuiltinLmTest.LinregType linregAlgo) {
		ExecMode platformOld = setExecMode(instType);
		String dml_test_name = TEST_NAME;
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			fullDMLScriptName = HOME + dml_test_name + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"), input("B"), input("Y0"), output("C")};
			double[][] A = TestUtils.ceil(getRandomMatrix(rows, cols, 0, 10, 1, 7));
			double[][] B = TestUtils.ceil(getRandomMatrix(10, 1, 0, 10, 1.0, 3));
			double[][] As = new double[rows][cols];
			double [] Ys = new double[rows];
			double Y[] = new double[rows];
			
			//Y = X %*% B
			for (int i = 0; i < rows; i++)
				for (int k = 0; k < cols; k++)
					Y[i] += A[i][k] * B[k][0];

			double AA[][] = new double[rows][cols+1];

			int value0 = 7;
			int value1 = 2;
			int coll0 = 5;
			int coll1 = 3;
			
			switch (test) {
				case 1:
					AA = modifyValue(A, Y,value0,coll0);
					break;
				case 2:
					AA = modifyValue(A, Y, value0, coll0);
					for(int i = 0;i<rows;i++){
						for(int j = 0; j < cols+1;j++){
							if(j == cols )
								Ys[i] = (int) AA[i][j];
							else
								As[i][j] = AA[i][j];
						}
					}
					AA = modifyValue(As,Ys,value1,coll0);
					break;
				case 3:
					AA = modifyValue(A, Y, value0, coll0);
					for(int i = 0;i<rows;i++){
						for(int j = 0; j < cols+1;j++){
							if(j == cols ){
								Ys[i] = (int) AA[i][j];
							}else{
								As[i][j] = AA[i][j];
							}
						}

					}
					AA = modifyValue(As,Ys,value1,coll1);
					break;
			}
			double[][] Y0 = new double[rows][1];
			for(int i = 0; i< rows;i++){
				Y0[i][0]= AA[i][10];
			}
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			writeInputMatrixWithMTD("Y0", Y0, true);
			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			double[][] ress = new double [5][20];
			for (Entry<MatrixValue.CellIndex, Double> a : dmlfile.entrySet()) {
				MatrixValue.CellIndex ci = a.getKey();
				ress[ci.row-1][ci.column-1] = a.getValue();
			}
			for(int i = 0; i < 5; i++){
				if(test == 1 ){
					if(ress[i][3] == value0 && ress[i][4] ==  coll0+1 && ress[i][12] == 0 && ress[i][13] == 0){
						System.out.print("Test passed!");
					}
				}else{
					if(((ress[i][3] == value0 | ress[i][3] == value1)  && (ress[i][4] ==  coll0+1 | ress[i][4] == coll1 +1)) && ((ress[i][12] == value0 | ress[i][12] == value1)
							&& (ress[i][13] == coll0+1 | ress[i][13] == coll1+1))){
						System.out.print("Test passed!");
					}
				}
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}
	
	private double[][] randomizeArray(double[][]y){
		Random rgen=new Random();
		for(int i=0; i<y.length; i++){
			int randomPosition=rgen.nextInt(y.length);
			double temp=y[i][0];
			y[i][0]=y[randomPosition][0];
			y[randomPosition][0]=temp;
		}
		return y;
	}

	private double[][] modifyValue(double[][] A, double[] Y, int value, int coll){
		int counter = 0;
		double nvec[][] = new double[rows][1];
		for (int i = 0; i < rows; i++) {
			if (A[i][coll] == value) {
				nvec[counter][0] = Y[i];
				counter++;
			}
		}
		double[][] y = new double[counter][1];
		for (int i = 0; i < counter; i++)
			y[i][0] = nvec[i][0];

		double[][] yy = randomizeArray(y);
		double AA [][] = new double[rows][cols + 1];
		counter = 0;

		for(int i = 0; i<rows; i++) {
			for(int j = 0; j < cols + 1;j++)
				AA[i][j] = (j == cols ) ? Y[i] : A[i][j];
			if(A[i][coll] == value) {  // this condition changes the values you choose
				AA[i][10] = yy[counter][0];
				counter++;
			}
		}
		return AA;
	}
}
