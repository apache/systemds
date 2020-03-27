/*
 * Copyright 2020 Graz University of Technology
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

package org.tugraz.sysds.test.applications;

import java.util.ArrayList;
import java.util.Collection;

import org.junit.After;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConstants;
import org.tugraz.sysds.test.TestConstants.CompressionType;
import org.tugraz.sysds.test.TestConstants.MatrixType;
import org.tugraz.sysds.test.TestConstants.SparsityType;

public class ApplicationTestBase extends AutomatedTestBase {

	protected static SparsityType[] usedSparsityTypes = new SparsityType[] { // Sparsities Used For testing.
		SparsityType.DENSE, //
		SparsityType.SPARSE, //
		// SparsityType.EMPTY
	};

	protected static CompressionType[] usedCompressionTypes = new CompressionType[] {CompressionType.LOSSLESS,
		// CompressionType.LOSSY,
	};

	protected static MatrixType[] usedMatrixType = new MatrixType[] { // Matrix Input sizes for testing
		MatrixType.SMALL,
		// MatrixType.LARGE,
		MatrixType.FEW_COL,
		MatrixType.FEW_ROW,
		// MatrixType.SINGLE_COL,
		// MatrixType.SINGLE_ROW,
		MatrixType.L_ROWS,
		MatrixType.XL_ROWS,
	};

	protected static ExecMode[] usedExecutionModes = new ExecMode[] { // The used execution modes
		ExecMode.SINGLE_NODE, ExecMode.HYBRID,
		// ExecMode.SPARK,
	};
	// ExecMode.values()

	protected int id; // Unique ID for each test case in parameterized classes

	protected int rows;
	protected int cols;
	protected double sparsity;

	protected ExecMode platformOld;
	protected boolean sparkConfigOld;

	public ApplicationTestBase(int id, SparsityType sparType, MatrixType matrixType, ExecMode newPlatform) {
		this.id = id;
		this.sparsity = TestConstants.getSparsityValue(sparType);
		this.rows = TestConstants.getNumberOfRows(matrixType);
		this.cols = TestConstants.getNumberOfColumns(matrixType);

		this.platformOld = rtplatform;
		rtplatform = newPlatform;
		this.sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		if(rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

	}

	@After
	public void teardown() {
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		int id = 0;
		for(SparsityType st : usedSparsityTypes) { // Test multiple Sparsities
			for(MatrixType mt : usedMatrixType) {
				for(ExecMode ex : usedExecutionModes) { // Test all Execution Modes.
					tests.add(new Object[] {id++, st, mt, ex});
				}
			}
		}
		return tests;
	}

	@Override
	public void setUp() {
	}

	@Override
	public void tearDown() {
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();

		builder.append("args: ");

		builder.append(String.format("%6s%5s", "Rows:", rows));
		builder.append(String.format("%6s%5s", "Cols:", cols));
		builder.append(String.format("%6s%4s", "Spar:", sparsity));

		return builder.toString();
	}
}
