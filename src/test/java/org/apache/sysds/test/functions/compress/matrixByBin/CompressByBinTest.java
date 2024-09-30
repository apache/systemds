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

package org.apache.sysds.test.functions.compress.matrixByBin;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.builtin.part1.BuiltinDistTest;
import org.junit.Assert;
import org.junit.Test;

public class CompressByBinTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(CompressByBinTest.class.getName());

	private final static String TEST_NAME = "compressByBins";
	private final static String TEST_DIR = "functions/compress/matrixByBin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDistTest.class.getSimpleName() + "/";

	private final static int rows = 1000;

	private final static int cols = 10;

	private final static int nbins = 10;

	// private final static int[] dVector = new int[cols];

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"X"}));
	}

	@Test
	public void testCompressBinsMatrixWidthCP() {
		runCompress(Types.ExecType.CP, ColumnEncoderBin.BinMethod.EQUI_WIDTH);
	}

	@Test
	public void testCompressBinsMatrixHeightCP() {
		runCompress(Types.ExecType.CP, ColumnEncoderBin.BinMethod.EQUI_HEIGHT);
	}

	@Test
	public void testCompressBinsFrameWidthCP() {
		runCompressFrame(Types.ExecType.CP, ColumnEncoderBin.BinMethod.EQUI_WIDTH);
	}

	@Test
	public void testCompressBinsFrameHeightCP() {
		runCompressFrame(Types.ExecType.CP, ColumnEncoderBin.BinMethod.EQUI_HEIGHT);
	}

	private void runCompress(Types.ExecType instType, ColumnEncoderBin.BinMethod binMethod) {
		Types.ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats","-args", input("X"),
				Boolean.toString(binMethod == ColumnEncoderBin.BinMethod.EQUI_WIDTH), output("meta"), output("res")};

			double[][] X = generateMatrixData(binMethod);
			writeInputMatrixWithMTD("X", X, true);

			runTest(null);

			checkMetaFile(DataConverter.convertToMatrixBlock(X), binMethod);

		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private void runCompressFrame(Types.ExecType instType, ColumnEncoderBin.BinMethod binMethod) {
		Types.ExecMode platformOld = setExecMode(instType);

		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-args", input("X"),
				Boolean.toString(binMethod == ColumnEncoderBin.BinMethod.EQUI_WIDTH), output("meta"), output("res")};

			Types.ValueType[] schema = new Types.ValueType[cols];
			Arrays.fill(schema, Types.ValueType.FP32);
			FrameBlock Xf = generateFrameData(binMethod, schema);
			writeInputFrameWithMTD("X", Xf, false, schema, Types.FileFormat.CSV);

			runTest(null);

			checkMetaFile(Xf, binMethod);

		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private double[][] generateMatrixData(ColumnEncoderBin.BinMethod binMethod) {
		double[][] X;
		if(binMethod == ColumnEncoderBin.BinMethod.EQUI_WIDTH) {
			// generate actual dataset
			X = getRandomMatrix(rows, cols, -100, 100, 1, 7);
			// make sure that bins in [-100, 100]
			for(int i = 0; i < cols; i++) {
				X[0][i] = -100;
				X[1][i] = 100;
			}
		}
		else if(binMethod == ColumnEncoderBin.BinMethod.EQUI_HEIGHT) {
			X = new double[rows][cols];
			for(int c = 0; c < cols; c++) {
				double[] vals = new Random().doubles(nbins).toArray();
				// Create one column
				for(int i = 0, j = 0; i < rows; i++) {
					X[i][c] = vals[j];
					if(i == ((j + 1) * (rows / nbins)))
						j++;
				}
			}
		}
		else
			throw new RuntimeException("Invalid binning method.");

		return X;
	}

	@SuppressWarnings("unchecked")
	private FrameBlock generateFrameData(ColumnEncoderBin.BinMethod binMethod, Types.ValueType[] schema) {
		FrameBlock Xf;
		if(binMethod == ColumnEncoderBin.BinMethod.EQUI_WIDTH) {
			Xf = TestUtils.generateRandomFrameBlock(1000, schema, 7);

			for(int i = 0; i < cols; i++) {
				Xf.set(0, i, -100);
				Xf.set(rows - 1, i, 100);
			}
		}
		else if(binMethod == ColumnEncoderBin.BinMethod.EQUI_HEIGHT) {
			Xf = new FrameBlock();
			for(int c = 0; c < schema.length; c++) {
				double[] vals = new Random().doubles(nbins).toArray();
				// Create one column
				Array<Float> f = (Array<Float>) ArrayFactory.allocate(Types.ValueType.FP32, rows);
				for(int i = 0, j = 0; i < rows; i++) {
					f.set(i, vals[j]);
					if(i == ((j + 1) * (rows / nbins)))
						j++;
				}
				Xf.appendColumn(f);
			}

		}
		else
			throw new RuntimeException("Invalid binning method.");

		return Xf;
	}

	private void checkMetaFile(CacheBlock<?> X, ColumnEncoderBin.BinMethod binningType) throws IOException {
		FrameBlock outputMeta = readDMLFrameFromHDFS("meta", Types.FileFormat.CSV);

		Assert.assertEquals(nbins, outputMeta.getNumRows());

		double[] binStarts = new double[nbins];
		double[] binEnds = new double[nbins];

		for(int c = 0; c < cols; c++) {
			if(binningType == ColumnEncoderBin.BinMethod.EQUI_WIDTH) {
				for(int i = -100, j = 0; i < 100; i += 20) {
					// check bin starts
					double binStart = Double.parseDouble(((String) outputMeta.getColumn(c).get(j)).split("·")[0]);
					Assert.assertEquals(i, binStart, 0.0);
					j++;
				}
			}
			else {
				binStarts[c] = Double.parseDouble(((String) outputMeta.getColumn(c).get(0)).split("·")[0]);
				binEnds[c] = Double.parseDouble(((String) outputMeta.getColumn(c).get(nbins - 1)).split("·")[1]);
			}
		}

		if(binningType == ColumnEncoderBin.BinMethod.EQUI_HEIGHT) {
			MatrixBlock mX = null;
			if(X instanceof FrameBlock) {
				mX = DataConverter.convertToMatrixBlock((FrameBlock) X);
			}
			else {
				mX = (MatrixBlock) X;
			}
			double[] colMins = mX.colMin().getDenseBlockValues();
			double[] colMaxs = mX.colMax().getDenseBlockValues();

			Assert.assertArrayEquals(colMins, binStarts, 0.0000000001);
			Assert.assertArrayEquals(colMaxs, binEnds, 0.0000000001);
		}
	}

}
