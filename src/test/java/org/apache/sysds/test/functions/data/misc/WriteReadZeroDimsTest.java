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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class WriteReadZeroDimsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ZeroDimDataWrite";
	private final static String TEST_NAME2 = "ZeroDimDataRead";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteReadZeroDimsTest.class.getSimpleName() + "/";
	
	private final static int rowsM = 1200;
	private final static int colsM = 1100;
	
	public enum Type{
		Zero_Rows,
		Zero_Cols,
	} 
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R1" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R2" }) );
	}

	@Test
	public void testZeroRowsTextCP() {
		runZeroDimsTest(Type.Zero_Rows, "text", ExecType.CP);
	}
	
	@Test
	public void testZeroColsTextCP() {
		runZeroDimsTest(Type.Zero_Cols, "text", ExecType.CP);
	}
	
	@Test
	public void testZeroRowsMmCP() {
		runZeroDimsTest(Type.Zero_Rows, "mm", ExecType.CP);
	}
	
	@Test
	public void testZeroColsMmCP() {
		runZeroDimsTest(Type.Zero_Cols, "mm", ExecType.CP);
	}
	
	@Test
	public void testZeroRowsCsvCP() {
		runZeroDimsTest(Type.Zero_Rows, "csv", ExecType.CP);
	}
	
	@Test
	public void testZeroColsCsvCP() {
		runZeroDimsTest(Type.Zero_Cols, "csv", ExecType.CP);
	}
	
	@Test
	public void testZeroRowsBinCP() {
		runZeroDimsTest(Type.Zero_Rows, "binary", ExecType.CP);
	}
	
	@Test
	public void testZeroColsBinCP() {
		runZeroDimsTest(Type.Zero_Cols, "binary", ExecType.CP);
	}

	@Test
	public void testZeroRowsTextSP() {
		runZeroDimsTest(Type.Zero_Rows, "text", ExecType.SPARK);
	}
	
	@Test
	public void testZeroColsTextSP() {
		runZeroDimsTest(Type.Zero_Cols, "text", ExecType.SPARK);
	}
	
	@Test
	public void testZeroRowsMmSP() {
		runZeroDimsTest(Type.Zero_Rows, "mm", ExecType.SPARK);
	}
	
	@Test
	public void testZeroColsMmSP() {
		runZeroDimsTest(Type.Zero_Cols, "mm", ExecType.SPARK);
	}
	
	@Test
	public void testZeroRowsCsvSP() {
		runZeroDimsTest(Type.Zero_Rows, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testZeroColsCsvSP() {
		runZeroDimsTest(Type.Zero_Cols, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testZeroRowsBinSP() {
		runZeroDimsTest(Type.Zero_Rows, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testZeroColsBinSP() {
		runZeroDimsTest(Type.Zero_Cols, "binary", ExecType.SPARK);
	}

	private void runZeroDimsTest( Type type, String format, ExecType et )
	{
		int rows = (type == Type.Zero_Rows) ? 0 : rowsM;
		int cols = (type == Type.Zero_Cols) ? 0 : colsM;
		
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try {
			//run write into format
			loadTestConfiguration(getTestConfiguration(TEST_NAME1));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", String.valueOf(rows),
				String.valueOf(cols), output("R1"), format};
			runTest(true, format.equals("csv"), null, -1);
			
			//run read from format
			if( !format.equals("csv") ) {
				loadTestConfiguration(getTestConfiguration(TEST_NAME2));
				HOME = SCRIPT_DIR + TEST_DIR;
				fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
				programArgs = new String[]{"-args", output("R1"), output("R2")};
				runTest(true, false, null, -1);
				
				//check overall result
				double expected = ((type == Type.Zero_Rows) ? colsM : rowsM) * 7;
				Assert.assertEquals(new Double(expected),
					readDMLMatrixFromHDFS("R2").get(new CellIndex(1,1)));
			}
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}