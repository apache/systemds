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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.runners.Parameterized.Parameters;

public class CompressedTestBase extends TestBase {

	protected static SparsityType[] usedSparsityTypes = new SparsityType[] { // Sparsity 0.9, 0.1, 0.01 and 0.0
		// SparsityType.DENSE,
		SparsityType.SPARSE, SparsityType.ULTRA_SPARSE, SparsityType.EMPTY
	};
	protected static ValueType[] usedValueTypes = new ValueType[] {
		// ValueType.RAND,
		ValueType.CONST, ValueType.RAND_ROUND, ValueType.OLE_COMPRESSIBLE, ValueType.RLE_COMPRESSIBLE,};

	protected static ValueRange[] usedValueRanges = new ValueRange[] {ValueRange.SMALL,
		// ValueRange.LARGE,
	};

	private static List<CompressionType> DDCOnly = new ArrayList<>();
	private static List<CompressionType> OLEOnly = new ArrayList<>();
	private static List<CompressionType> RLEOnly = new ArrayList<>();

	static {
		DDCOnly.add(CompressionType.DDC);
		OLEOnly.add(CompressionType.OLE);
		RLEOnly.add(CompressionType.RLE);
	}

	private static final int compressionSeed = 7;

	protected static CompressionSettings[] usedCompressionSettings = new CompressionSettings[] {
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setAllowSharedDDCDictionary(false)
			.setSeed(compressionSeed).setValidCompressions(DDCOnly).setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setAllowSharedDDCDictionary(true)
			.setSeed(compressionSeed).setValidCompressions(DDCOnly).setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed).setValidCompressions(OLEOnly)
			.setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed).setValidCompressions(RLEOnly)
			.setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(1.0).setSeed(compressionSeed).setInvestigateEstimate(true)
			.create()};

	protected static MatrixTypology[] usedMatrixTypology = new MatrixTypology[] { // Selected Matrix Types
		MatrixTypology.SMALL,
		// MatrixTypology.FEW_COL,
		MatrixTypology.FEW_ROW,
		// MatrixTypology.LARGE,
		// MatrixTypology.SINGLE_COL,
		// MatrixTypology.SINGLE_ROW,
		MatrixTypology.L_ROWS,
		// MatrixTypology.XL_ROWS,
	};

	// Compressed Block
	protected MatrixBlock cmb;

	// Decompressed Result
	protected MatrixBlock cmbDeCompressed;
	protected double[][] deCompressed;

	// Threads
	protected int k = 1;

	protected int sampleTolerance = 1024;

	public CompressedTestBase(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettings compSettings, MatrixTypology MatrixTypology) {
		super(sparType, valType, valueRange, compSettings, MatrixTypology);
		// System.out.println("HERE !");
		try {

			cmb = CompressedMatrixBlockFactory.compress(mb, k, compressionSettings);

			if(cmb instanceof CompressedMatrixBlock) {
				cmbDeCompressed = ((CompressedMatrixBlock) cmb).decompress();
				if(cmbDeCompressed != null) {

					deCompressed = DataConverter.convertToDoubleMatrix(cmbDeCompressed);
				}
			}
			else {
				cmbDeCompressed = null;
				deCompressed = null;
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			// throw new RuntimeException(
			// "CompressionTest Init failed with settings: " + this.toString() + "\n" + e.getMessage(), e);
			assertTrue("\nCompressionTest Init failed with settings: " + this.toString(), false);
		}

	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		for(SparsityType st : usedSparsityTypes) {
			for(ValueType vt : usedValueTypes) {
				for(ValueRange vr : usedValueRanges) {
					for(CompressionSettings cs : usedCompressionSettings) {
						for(MatrixTypology mt : usedMatrixTypology) {
							tests.add(new Object[] {st, vt, vr, cs, mt,});

						}
					}
				}
			}
		}

		return tests;
	}
}
