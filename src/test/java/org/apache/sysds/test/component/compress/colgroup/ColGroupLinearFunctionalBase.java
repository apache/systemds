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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupLinearFunctional;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public abstract class ColGroupLinearFunctionalBase {

	protected static final Log LOG = LogFactory.getLog(ColGroupLinearFunctionalBase.class.getName());
	private final static Random random = new Random();
	protected final AColGroup base;
	protected final ColGroupLinearFunctional lin;
	protected final AColGroup baseLeft;
	protected final int nRowLeft;
	protected final int nColLeft;

	protected final int nRowRight;
	protected final int nColRight;

	protected final AColGroup cgLeft;
	protected final ColGroupUncompressed cgRight;
	protected final int nRow;
	protected final double tolerance;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			addLinCases(tests);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public ColGroupLinearFunctionalBase(AColGroup base, ColGroupLinearFunctional lin, AColGroup baseLeft,
		AColGroup cgLeft, int nRowLeft, int nColLeft, int nRowRight, int nColRight, ColGroupUncompressed cgRight,
		double tolerance) {
		if(lin.getNumCols() != base.getNumCols())
			fail("Linearly compressed ColGroup and Base ColGroup must have same number of columns");

		if(nRowLeft != lin.getNumRows())
			fail("Transposed left ColGroup and center ColGroup (`lin`) must have compatible dimensions");

		int[] colIndices = lin.getColIndices();
		if(colIndices[colIndices.length - 1] > nRowRight)
			fail("Right ColGroup must have at least as many rows as the largest column index of center ColGroup (`lin`)");

		this.base = base;
		this.lin = lin;
		this.baseLeft = baseLeft;
		this.nRowLeft = nRowLeft;
		this.nColLeft = nColLeft;
		this.nRowRight = nRowRight;
		this.nColRight = nColRight;
		this.cgLeft = cgLeft;
		this.cgRight = cgRight;
		this.tolerance = tolerance;
		this.nRow = lin.getNumRows();
	}

	protected static void addLinCases(ArrayList<Object[]> tests) {
		double[][] data = new double[][] {{1, 2, 3, 4, 5}, {-4, 2, 8, 14, 20}};
		double[][] dataRight = new double[][] {{1, -2, 23, 7}, {4, 11, -10, -2}};
		double[][] dataLeft = new double[][] {{8, 3, 7, 12, -3}, {-1, 8, 4, -2, -2}, {3, 4, 2, 0, -1}};
		int[] colIndexesLeft = new int[] {0, 2};

		double[][] dataLeftCompressed = new double[][] {{8, 4, 0, -4, -8}, {-1, 0, 1, 2, 3}};
		int[] colIndexesLeftCompressed = new int[] {0};

		tests
			.add(createInitParams(data, true, null, dataLeft, true, colIndexesLeft, false, dataRight, true, null, 0.001));

		tests.add(createInitParams(data, true, null, dataLeftCompressed, true, colIndexesLeftCompressed, true, dataRight,
			true, null, 0.001));

		tests.add(createInitParams(new double[][] {{1, 2, 3, 4, 5}}, true, null, null, true, null, true, dataRight, true,
			null, 0.001));

		tests.add(createInitParams(new double[][] {{1, 2, 3, 4, 5}}, true, null, null, true, null, true, dataRight, true,
			null, 0.001));

		tests.add(createInitParams(new double[][] {{1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}, {4, 2, 4, 2, 4}}, true,
			new int[] {0, 1}, null, true, null, true, dataRight, true, null, 0.001));

		tests.add(createInitParams(new double[][] {{1, 2, 3, 4, 5}, {-1, -2, -3, -4, -5}}, true, null, null, true, null,
			true, dataRight, true, null, 0.001));

		double[][] randomData = generateTestMatrixLinear(80, 100, -100, 100, -1, 1, 42);
		double[][] randomDataLeft = generateTestMatrixLinear(80, 50, -100, 100, -1, 1, 43);
		double[][] randomDataRight = generateTestMatrixLinear(100, 500, -100, 100, -1, 1, 44);

		tests.add(createInitParams(randomData, false, null, randomDataLeft, false, null, true, randomDataRight, true,
			null, 0.001));
	}

	protected static Object[] createInitParams(double[][] data, boolean isTransposed, int[] colIndexes,
		double[][] dataLeft, boolean transposedLeft, int[] colIndexesLeft, boolean linCompressLeft, double[][] dataRight,
		boolean transposedRight, int[] colIndexesRight, double tolerance) {
		if(dataLeft == null)
			dataLeft = data;

		// int nRow = isTransposed ? data[0].length : data.length;
		int nCol = isTransposed ? data.length : data[0].length;
		int nRowLeft = transposedLeft ? dataLeft[0].length : dataLeft.length;
		int nColLeft = transposedLeft ? dataLeft.length : dataLeft[0].length;
		int nRowRight = transposedRight ? dataRight[0].length : dataRight.length;
		int nColRight = transposedRight ? dataRight.length : dataRight[0].length;

		if(colIndexes == null)
			colIndexes = Util.genColsIndices(nCol);

		if(colIndexesLeft == null)
			colIndexesLeft = Util.genColsIndices(nColLeft);

		if(colIndexesRight == null)
			colIndexesRight = Util.genColsIndices(nColRight);

		return new Object[] {cgUncompressed(data, colIndexes, isTransposed),
			cgLinCompressed(data, colIndexes, isTransposed), cgUncompressed(dataLeft, colIndexesLeft, transposedLeft),
			linCompressLeft ? cgLinCompressed(dataLeft, colIndexesLeft, transposedLeft) : cgUncompressed(dataLeft,
				colIndexesLeft, transposedLeft),
			nRowLeft, nColLeft, nRowRight, nColRight, cgUncompressed(dataRight, colIndexesRight, transposedRight),
			tolerance};
	}

	protected static AColGroup cgUncompressed(double[][] data, int[] colIndexes, boolean isTransposed) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		return createColGroup(mbt, colIndexes, isTransposed, AColGroup.CompressionType.UNCOMPRESSED);
	}

	protected static AColGroup cgLinCompressed(double[][] data, boolean isTransposed) {
		final int numCols = isTransposed ? data.length : data[0].length;
		return cgLinCompressed(data, Util.genColsIndices(numCols), isTransposed);
	}

	protected static AColGroup cgLinCompressed(double[][] data, int[] colIndexes, boolean isTransposed) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		return createColGroup(mbt, colIndexes, isTransposed, AColGroup.CompressionType.LinearFunctional);
	}

	public static AColGroup createColGroup(MatrixBlock mbt, int[] colIndexes, boolean isTransposed,
		AColGroup.CompressionType cgType) {
		CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(cgType)).create();
		cs.transposed = isTransposed;

		final CompressedSizeInfoColGroup cgi = new CompressedSizeEstimatorExact(mbt, cs).getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		return ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);
	}

	public static double[] generateLinearColumn(double intercept, double slope, int length) {
		double[] result = new double[length];
		for(int i = 0; i < length; i++) {
			result[i] = intercept + slope * (i + 1);
		}

		return result;
	}

	public static double[][] generateTestMatrixLinear(int rows, int cols, double minIntercept, double maxIntercept,
		double minSlope, double maxSlope, long seed) {
		double[][] coefficients = generateRandomInterceptsSlopes(cols, minIntercept, maxIntercept, minSlope, maxSlope,
			seed);
		return generateTestMatrixLinearColumns(rows, cols, coefficients[0], coefficients[1]);
	}

	public static double[][] generateRandomInterceptsSlopes(int cols, double minIntercept, double maxIntercept,
		double minSlope, double maxSlope, long seed) {

		double[] intercepts = new double[cols];
		double[] slopes = new double[cols];

		random.setSeed(seed);
		for(int j = 0; j < cols; j++) {
			intercepts[j] = minIntercept + random.nextDouble() * (maxIntercept - minIntercept);
			slopes[j] = minSlope + random.nextDouble() * (maxSlope - minSlope);
		}

		return new double[][] {intercepts, slopes};
	}

	public static double[][] generateTestMatrixLinearColumns(int rows, int cols, double[] intercepts, double[] slopes) {
		if(intercepts.length != slopes.length || intercepts.length != cols)
			fail("Intercepts and slopes array must both have length `cols`");

		double[][] data = new double[rows][cols];

		for(int j = 0; j < cols; j++) {
			double[] linCol = generateLinearColumn(intercepts[j], slopes[j], rows);
			for(int i = 0; i < rows; i++) {
				data[i][j] = linCol[i];
			}
		}

		return data;
	}

	protected double[] getValues(AColGroup cg) {
		MatrixBlock mb = new MatrixBlock(nRow, cg.getNumCols(), false);
		mb.allocateDenseBlock();
		cg.decompressToDenseBlock(mb.getDenseBlock(), 0, nRow);
		return mb.getDenseBlockValues();
	}

}
