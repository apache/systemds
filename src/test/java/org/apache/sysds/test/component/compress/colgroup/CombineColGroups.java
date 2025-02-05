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
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CombineColGroups {
	protected static final Log LOG = LogFactory.getLog(CombineColGroups.class.getName());

	/** Uncompressed ground truth */
	final MatrixBlock mb;
	/** ColGroup 1 */
	final AColGroup a;
	/** ColGroup 2 */
	final AColGroup b;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			addTwoCols(tests, 100, 3);
			addTwoCols(tests, 1000, 3);
			// addSingleVSMultiCol(tests, 100, 3, 1, 3);
			// addSingleVSMultiCol(tests, 100, 3, 3, 4);
			addSingleVSMultiCol(tests, 1000, 3, 1, 3, 1.0);
			addSingleVSMultiCol(tests, 1000, 3, 3, 4, 1.0);
			addSingleVSMultiCol(tests, 1000, 3, 3, 1, 1.0);
			addSingleVSMultiCol(tests, 1000, 2, 1, 10, 0.05);
			addSingleVSMultiCol(tests, 1000, 2, 10, 10, 0.05);
			addSingleVSMultiCol(tests, 1000, 2, 10, 1, 0.05);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public CombineColGroups(MatrixBlock mb, AColGroup a, AColGroup b) {
		this.mb = mb;
		this.a = a;
		this.b = b;

		CompressedMatrixBlock.debug = true;
	}

	@Test
	public void combine() {
		try {
			AColGroup c = a.combine(b, mb.getNumRows());
			MatrixBlock ref = new MatrixBlock(mb.getNumRows(), mb.getNumColumns(), false);
			ref.allocateDenseBlock();
			c.decompressToDenseBlock(ref.getDenseBlock(), 0, mb.getNumRows());
			ref.recomputeNonZeros();
			String errMessage = a.getClass().getSimpleName() + ": " + a.getColIndices() + " -- "
				+ b.getClass().getSimpleName() + ": " + b.getColIndices();

			TestUtils.compareMatricesBitAvgDistance(mb, ref, 0, 0, errMessage);
		}
		catch(NotImplementedException | DMLCompressionException e) {
			// allowed
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static void addTwoCols(ArrayList<Object[]> tests, int nRow, int distinct) {
		MatrixBlock mb = TestUtils.ceil(//
			TestUtils.generateTestMatrixBlock(nRow, 2, 0, distinct, 1.0, 231));

		List<AColGroup> c1s = getGroups(mb, ColIndexFactory.createI(0));
		List<AColGroup> c2s = getGroups(mb, ColIndexFactory.createI(1));

		for(int i = 0; i < c1s.size(); i++) {
			for(int j = 0; j < c2s.size(); j++) {
				tests.add(new Object[] {mb, c1s.get(i), c2s.get(j)});
			}
		}
	}

	private static void addSingleVSMultiCol(ArrayList<Object[]> tests, int nRow, int distinct, int nColL, int nColR,
		double sparsity) {
		MatrixBlock mb = TestUtils.ceil(//
			TestUtils.generateTestMatrixBlock(nRow, nColL + nColR, 0, distinct, sparsity, 231));

		List<AColGroup> c1s = getGroups(mb, ColIndexFactory.create(nColL));
		List<AColGroup> c2s = getGroups(mb, ColIndexFactory.create(nColL, nColR + nColL));

		for(int i = 0; i < c1s.size(); i++) {
			for(int j = 0; j < c2s.size(); j++) {
				tests.add(new Object[] {mb, c1s.get(0), c2s.get(0)});
			}
		}
	}

	private static List<AColGroup> getGroups(MatrixBlock mb, IColIndex cols) {
		final CompressionSettings cs = new CompressionSettingsBuilder().create();

		final int nRow = mb.getNumColumns();
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		final EstimationFactors f = new EstimationFactors(nRow, nRow, mb.getSparsity());
		es.add(new CompressedSizeInfoColGroup(cols, f, 312152, CompressionType.DDC));
		es.add(new CompressedSizeInfoColGroup(cols, f, 321521, CompressionType.RLE));
		es.add(new CompressedSizeInfoColGroup(cols, f, 321452, CompressionType.SDC));
		es.add(new CompressedSizeInfoColGroup(cols, f, 325151, CompressionType.UNCOMPRESSED));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);
		return ColGroupFactory.compressColGroups(mb, csi, cs);
	}
}
