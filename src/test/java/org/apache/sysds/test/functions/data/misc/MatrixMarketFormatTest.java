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

package org.apache.sysds.test.functions.data.misc;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM.MMField;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM.MMFormat;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM.MMSymmetry;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;

import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class MatrixMarketFormatTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "MatrixMarketFormat";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MatrixMarketFormatTest.class.getSimpleName() + "/";
	
	private final static int dim = 1200;
	private final static double sparsity = 0.1;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R", "C" }) );
	}

	@Test
	public void testMMCooRealGeneralCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.GENERAL, ExecType.CP);
	}
	
	@Test
	public void testMMCooRealGeneralSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.GENERAL, ExecType.SPARK);
	}
	
	@Test
	public void testMMCooRealSymmetricCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.SYMMETRIC, ExecType.CP);
	}
	
	@Test
	public void testMMCooRealSymmetricSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.SYMMETRIC, ExecType.SPARK);
	}

//	@Test
//	public void testMMCooRealSkewSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.SKEW_SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMCooRealSkewSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.REAL, MMSymmetry.SKEW_SYMMETRIC, ExecType.SPARK);
//	}
	
	@Test
	public void testMMCooIntegerGeneralCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.GENERAL, ExecType.CP);
	}
	
	@Test
	public void testMMCooIntegerGeneralSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.GENERAL, ExecType.SPARK);
	}
	
	@Test
	public void testMMCooIntegerSymmetricCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.SYMMETRIC, ExecType.CP);
	}
	
	@Test
	public void testMMCooIntegerSymmetricSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.SYMMETRIC, ExecType.SPARK);
	}

//	@Test
//	public void testMMCooIntegerSkewSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.SKEW_SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMCooIntegerSkewSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.INTEGER, MMSymmetry.SKEW_SYMMETRIC, ExecType.SPARK);
//	}
	
	@Test
	public void testMMCooPatternGeneralCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.PATTERN, MMSymmetry.GENERAL, ExecType.CP);
	}
	
	@Test
	public void testMMCooPatternGeneralSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.PATTERN, MMSymmetry.GENERAL, ExecType.SPARK);
	}
	
	@Test
	public void testMMCooPatternSymmetricCP() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.PATTERN, MMSymmetry.SYMMETRIC, ExecType.CP);
	}
	
	@Test
	public void testMMCooPatternSymmetricSp() {
		runMatrixMarketFormatTest(MMFormat.COORDINATE, MMField.PATTERN, MMSymmetry.SYMMETRIC, ExecType.SPARK);
	}

//	@Test
//	public void testMMArrRealGeneralCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.GENERAL, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrRealGeneralSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.GENERAL, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testMMArrRealSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrRealSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.SYMMETRIC, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testMMArrRealSkewSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.SKEW_SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrRealSkewSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.REAL, MMSymmetry.SKEW_SYMMETRIC, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testMMArrIntegerGeneralCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.GENERAL, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrIntegerGeneralSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.GENERAL, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testMMArrIntegerSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrIntegerSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.SYMMETRIC, ExecType.SPARK);
//	}
//	
//	@Test
//	public void testMMArrIntegerSkewSymmetricCP() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.SKEW_SYMMETRIC, ExecType.CP);
//	}
//	
//	@Test
//	public void testMMArrIntegerSkewSymmetricSp() {
//		runMatrixMarketFormatTest(MMFormat.ARRAY, MMField.INTEGER, MMSymmetry.SKEW_SYMMETRIC, ExecType.SPARK);
//	}

	private void runMatrixMarketFormatTest(MMFormat fmt, MMField field, MMSymmetry symmetry, ExecType et)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.SINGLE_NODE; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullRScriptName = HOME + TEST_NAME + ".R";
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), output("R"), output("C") };
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				input("X") + " " + expected("R") + " " + expected("C");
			
			generateAndWriteMMInput(input("X"), fmt, field, symmetry);
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare row and column aggregates
			TestUtils.compareMatrices(readDMLMatrixFromHDFS("R"),
				readRMatrixFromFS("R"), 1e-10, "Stat-DML", "Stat-R");
			TestUtils.compareMatrices(readDMLMatrixFromHDFS("C"),
				readRMatrixFromFS("C"), 1e-10, "Stat-DML", "Stat-R");
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	private static void generateAndWriteMMInput(String fname, MMFormat fmt, MMField field, MMSymmetry symmetry) 
		throws IOException 
	{
		int rows = dim;
		int cols = (symmetry==MMSymmetry.GENERAL) ? dim/3 : dim;
		MatrixBlock tmp = MatrixBlock.randOperations(
			rows, cols, sparsity, -10, 10, "uniform", 7);
		
		String header = "%%MatrixMarket matrix " + fmt.toString() + " " 
			+ field.toString() + " " + symmetry.toString() + "\n";
		String meta = rows + " " + cols + ((fmt == MMFormat.COORDINATE) ?
			" " + tmp.getNonZeros() : "") + "\n";
		
		Path path = new Path( fname );
		FileSystem fs = IOUtilFunctions.getFileSystem(path);
		
		try( BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(path,true))) )
		{
			br.write(header);
			br.write(meta);
			
			if( fmt == MMFormat.ARRAY ) {
				for(int j=0; j<tmp.getNumColumns(); j++) {
					int bi = (symmetry == MMSymmetry.GENERAL) ? 0 :
						(symmetry == MMSymmetry.SYMMETRIC) ? j : j+1;
					for(int i=bi; i<tmp.getNumRows(); i++) {
						double val = tmp.quickGetValue(i, j);
						br.write(String.valueOf((field == MMField.INTEGER) ?
							(int) val : val) + "\n" );
					}
				}
			}
			else { //COORDINATE
				if( tmp.isInSparseFormat() ) {
					StringBuilder sb = new StringBuilder();
					Iterator<IJV> iter = tmp.getSparseBlockIterator();
					while( iter.hasNext() ) {
						IJV cell = iter.next();
						if( (symmetry == MMSymmetry.SYMMETRIC && cell.getJ() > cell.getI())
							|| (symmetry == MMSymmetry.SKEW_SYMMETRIC && cell.getJ() >= cell.getI()))
							continue;
						sb.append(cell.getI()+1);
						sb.append(' ');
						sb.append(cell.getJ()+1);
						if( field != MMField.PATTERN ) {
							sb.append(' ');
							sb.append((field == MMField.INTEGER) ? 
								String.valueOf((int) cell.getV()) : String.valueOf(cell.getV()));
						}
						sb.append('\n');
						br.write( sb.toString() ); //same as append
						sb.setLength(0); 
					}
				}
				else {
					//always sparse in above used setup
					throw new NotImplementedException();
				}
			}
		}
	}
}
