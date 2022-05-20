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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Basic idea is that we specify a list of compression schemes that a input is allowed to be compressed into. The test
 * verify that these all does the same on a given input. Base on the api for a columnGroup.
 */
@RunWith(value = Parameterized.class)
public abstract class ColGroupBase {
	protected static final Log LOG = LogFactory.getLog(ColGroupBase.class.getName());

	protected final int nRow;
	protected final int maxCol;
	protected final AColGroup base;
	protected final AColGroup other;
	protected final double tolerance;

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {

			addConstCases(tests);
			addVariations(tests);
			addSingle(tests);
			addSingleZeros(tests);
			addSingleSparseZeros(tests);
			addUncompressedColGroup(tests);
			addSDC_CONST_BeforeAndAfter(tests);
			addOverCharLong(tests);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public ColGroupBase(AColGroup base, AColGroup other, int nRow) {
		this.base = base;
		this.other = other;
		this.tolerance = 0;
		this.nRow = nRow;
		this.maxCol = Arrays.stream(base.getColIndices()).max().getAsInt() + 1;
	}

	protected AColGroup serializeAndBack(AColGroup g) {
		try {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			ColGroupIO.writeGroups(fos, Collections.singletonList(g));
			DataInputStream fis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
			return ColGroupIO.readGroups(fis, nRow).get(0);
		}
		catch(IOException e) {
			throw new RuntimeException("Error in io", e);
		}
	}

	protected static AColGroup cConst(double[] v) {
		return ColGroupConst.create(v);
	}

	protected static AColGroup cConst(int[] i, double[] v) {
		return ColGroupConst.create(i, v);
	}

	protected MatrixBlock sparseMB(int nCol) {
		return sparseMB(nRow, nCol);
	}

	protected static MatrixBlock sparseMB(int nRow, int nCol) {
		MatrixBlock t = new MatrixBlock(nRow, nCol, true);
		t.allocateSparseRowsBlock();
		return t;
	}

	protected MatrixBlock denseMB(int nCol) {
		MatrixBlock t = new MatrixBlock(nRow, nCol, false);
		t.allocateDenseBlock();
		return t;
	}

	protected static void compare(MatrixBlock m1, MatrixBlock m2) {
		if(m1.isEmpty())
			m1.recomputeNonZeros();
		if(m2.isEmpty())
			m2.recomputeNonZeros();
		TestUtils.compareMatricesBitAvgDistance(m1, m2, 0, 0, "");
	}

	protected static void addAll(ArrayList<Object[]> tests, int nRow, int nVal, double sparsity, int seed) {
		final MatrixBlock mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, nRow, 1, nVal + 1, sparsity, seed));
		addAll(tests, mbt, nVal);
	}

	protected static void addAll(ArrayList<Object[]> tests, int nRow, int nCol, int nVal, double sparsity, int seed) {
		final MatrixBlock mbt = TestUtils
			.ceil(TestUtils.generateTestMatrixBlock(nCol, nRow, 1, nVal + 1, sparsity, seed));
		int[] cols = Util.genColsIndices(nCol);
		addAll(tests, mbt, cols);
	}

	protected static void addAll(ArrayList<Object[]> tests, MatrixBlock mbt, int nVal) {
		final int[] colZero = new int[] {0};
		addAll(tests, mbt, colZero);
	}

	protected static void addAll(ArrayList<Object[]> tests, MatrixBlock mbt, int[] cols) {
		final CompressionSettings cs = new CompressionSettingsBuilder().create();
		cs.transposed = true;
		addAll(tests, mbt, cols, cs);

	}

	protected static void addAll(ArrayList<Object[]> tests, MatrixBlock mbt, int[] cols, CompressionSettings cs) {

		try {

			final int nRow = mbt.getNumColumns();

			final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
			final EstimationFactors f = new EstimationFactors(nRow, nRow, mbt.getSparsity());
			es.add(new CompressedSizeInfoColGroup(cols, f, 312152, CompressionType.DDC));
			es.add(new CompressedSizeInfoColGroup(cols, f, 321521, CompressionType.RLE));
			es.add(new CompressedSizeInfoColGroup(cols, f, 321452, CompressionType.SDC));
			es.add(new CompressedSizeInfoColGroup(cols, f, 325151, CompressionType.UNCOMPRESSED));
			final CompressedSizeInfo csi = new CompressedSizeInfo(es);
			final List<AColGroup> comp = ColGroupFactory.compressColGroups(mbt, csi, cs);

			final ScalarOperator sop = new RightScalarOperator(Modulus.getFnObject(), 2);
			for(int i = 0; i < comp.size(); i++) {
				if(i < comp.size() - 1)
					tests.add(new Object[] {comp.get(i), comp.get(i + 1), nRow});
				if(comp.get(i) instanceof ColGroupDDC) {
					AColGroup sp = ((ColGroupDDC) comp.get(i)).sparsifyFOR();
					if(sp != comp.get(i)) {
						tests.add(new Object[] {comp.get(i), sp, nRow});
						tests.add(new Object[] {comp.get(i).scalarOperation(sop), sp.scalarOperation(sop), nRow});
					}
				}
				else if(comp.get(i) instanceof ColGroupSDC) {
					AColGroup sp = ((ColGroupSDC) comp.get(i)).sparsifyFOR();
					if(sp != comp.get(i)) {
						tests.add(new Object[] {comp.get(i), sp, nRow});
						tests.add(new Object[] {comp.get(i).scalarOperation(sop), sp.scalarOperation(sop), nRow});
					}
				}
				else if(comp.get(i) instanceof ColGroupUncompressed) {
					AColGroup ucm = ColGroupUncompressed.create(cols, LibMatrixReorg.transpose(mbt), false);
					tests.add(new Object[] {comp.get(i), ucm, nRow});
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected static void addSingle(ArrayList<Object[]> tests) {
		MatrixBlock mb = new MatrixBlock(100, 100, true);
		MatrixBlock mv = new MatrixBlock(100, 1, true);
		mv.quickSetValue(4, 0, 1);
		mv.quickSetValue(45, 0, 2);
		mv.quickSetValue(2, 0, 3);
		mv.quickSetValue(66, 0, 4);
		mv.quickSetValue(99, 0, 5);
		BinaryOperator bop = new BinaryOperator(Plus.getPlusFnObject(), 1);
		MatrixBlock mbr = mb.binaryOperations(bop, mv, null);
		MatrixBlock mbrc = new MatrixBlock();
		mbrc.copy(mbr);

		for(int i = 0; i < 100; i++)
			mbrc.quickSetValue(0, i, 100);

		mbrc.recomputeNonZeros();
		addAll(tests, mbrc, Util.genColsIndices(10));
		addAll(tests, mbrc, Util.genColsIndices(100));

		MatrixBlock mbr2 = new MatrixBlock();
		mbr2.copy(mbr);

		for(int j : new int[] {1, 5, 23, 51, 62})
			for(int i = 0; i < 100; i++)
				mbr2.quickSetValue(j, i, 100 * j);

		mbr2.recomputeNonZeros();
		addAll(tests, mbr2, Util.genColsIndices(10));
		addAll(tests, mbr2, Util.genColsIndices(100));

		MatrixBlock mbr3 = mbr2.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, mbr3, Util.genColsIndices(10));
		addAll(tests, mbr3, Util.genColsIndices(100));
	}

	protected static void addSingleZeros(ArrayList<Object[]> tests) {
		try {

			MatrixBlock mb = new MatrixBlock(100, 100, true);

			for(int j : new int[] {1, 4, 44, 87})
				for(int i = 0; i < 100; i++)
					mb.quickSetValue(i, j, 100);

			mb.recomputeNonZeros();
			addAll(tests, mb, Util.genColsIndices(10));
			addAll(tests, mb, Util.genColsIndices(100));
		}

		catch(Exception e) {
			e.printStackTrace();
			fail("Failed constructing zeros");
		}
	}

	protected static void addSingleSparseZeros(ArrayList<Object[]> tests) {
		try {

			MatrixBlock mb = new MatrixBlock(100, 100, true);

			for(int j : new int[] {0, 4, 44, 87})
				for(int i : new int[] {0, 4, 13, 24, 56, 92})
					mb.quickSetValue(i, j, 100);

			mb.recomputeNonZeros();
			addAll(tests, mb, Util.genColsIndices(1));
			addAll(tests, mb, Util.genColsIndices(10));
			addAll(tests, mb, Util.genColsIndices(100));

			MatrixBlock mb2 = new MatrixBlock(100, 100, true);

			for(int j : new int[] {44, 87})
				for(int i : new int[] {56, 92})
					mb2.quickSetValue(i, j, 100);

			mb2.recomputeNonZeros();
			addAll(tests, mb2, Util.genColsIndices(100));

			MatrixBlock mb3 = new MatrixBlock(100, 100, true);

			for(int j : new int[] {1, 2})
				for(int i : new int[] {1, 2})
					mb3.quickSetValue(i, j, 100);

			mb3.recomputeNonZeros();
			addAll(tests, mb3, Util.genColsIndices(100));
		}

		catch(Exception e) {
			e.printStackTrace();
			fail("Failed constructing zeros");
		}
	}

	protected static void addVariations(ArrayList<Object[]> tests) {

		for(int n : new int[] {40, 100}) {

			// Create more tests.
			addAll(tests, n, 10, 0.7, 1342);
			addAll(tests, n, 2, 0.7, 1342);
			addAll(tests, n, 1, 0.7, 1342);
			addAll(tests, n, 1, 0.3, 1342);
			addAll(tests, n, 2, 1.0, 1342);

			addAll(tests, n, 3, 2, 1.0, 1342);
			addAll(tests, n, 5, 2, 1.0, 1342);
			addAll(tests, n, 10, 2, 1.0, 1342);

			addAll(tests, n, 3, 2, 0.2, 1342);
			addAll(tests, n, 5, 2, 0.2, 1342);
			addAll(tests, n, 10, 2, 0.2, 1342);

			addAll(tests, n, 15, 1, 0.1, 325);
			addAll(tests, n, 32, 1, 0.1, 1342);
			addAll(tests, n, 100, 2, 0.2, 1342);

			// Empty
			addAll(tests, n, 13, 2, 0.0, 1342);
			addAll(tests, n, 300, 2, 0.0, 1342);

		}
	}

	protected static void addConstCases(ArrayList<Object[]> tests) {
		final int nRowS = 40;
		tests.add(new Object[] {cConst(new double[] {0}), cConst(new double[] {0}), nRowS});
		tests.add(new Object[] {cConst(new double[] {23}), cConst(new double[] {23}), nRowS});
		tests.add(new Object[] {cConst(new double[] {0, 23}), cConst(new double[] {0, 23}), nRowS});
		tests.add(new Object[] {cConst(new double[] {0, 23, 1}), cConst(new double[] {0, 23, 1}), nRowS});

		int[] oneCol = new int[] {1};
		tests.add(new Object[] {cConst(oneCol, new double[] {1}), cConst(oneCol, new double[] {1}), nRowS});

	}

	protected static void addUncompressedColGroup(ArrayList<Object[]> tests) {
		MatrixBlock m = new MatrixBlock(100, 10, true);

		tests.add(new Object[] {ColGroupUncompressed.create(m), ColGroupEmpty.create(10), 100});
	}

	protected static void addSDC_CONST_BeforeAndAfter(ArrayList<Object[]> tests) {
		int[] cols = Util.genColsIndices(10);
		MatrixBlock mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10, 100, 1, 4, 0.2, 1342));
		MatrixBlock empty = new MatrixBlock(10, 100, 0.0);
		MatrixBlock n = empty.append(mbt).append(empty);

		final CompressionSettings cs = new CompressionSettingsBuilder().setSortValuesByLength(false).create();
		cs.transposed = true;

		MatrixBlock m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);

		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(10, 100, -4, -1, 0.2, 1342));
		// empty = new MatrixBlock(10, 100, 0.0);
		n = empty.append(mbt).append(empty);
		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);

		cols = Util.genColsIndices(1);

		// SDC Single
		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, 1, 1, 0.2, 1342));
		empty = new MatrixBlock(1, 100, 0.0);
		n = empty.append(mbt).append(empty);

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);

		// SDC Single minus 1
		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, -1, -1, 0.2, 1342));
		n = empty.append(mbt).append(empty);
		addAll(tests, n.append(n, false), Util.genColsIndices(2));

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);
		addAll(tests, m.append(m, false), Util.genColsIndices(2));

		// SDC minus range
		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, -3, -1, 0.2, 1342));
		n = empty.append(mbt).append(empty);
		addAll(tests, n.append(n, false), Util.genColsIndices(2));

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);
		addAll(tests, m.append(m, false), Util.genColsIndices(2));

		// SDC minus range add More
		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, -3, -1, 0.2, 1342));
		n = empty.append(mbt).append(empty);
		addAll(tests, n.append(n, false), Util.genColsIndices(2));

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 4), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);
		addAll(tests, m.append(m, false), Util.genColsIndices(2));

		// SDC add in between
		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, -3, -1, 0.2, 1342));
		n = empty.append(mbt).append(empty);
		addAll(tests, n.append(n, false), Util.genColsIndices(2));

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 2), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);
		addAll(tests, m.append(m, false), Util.genColsIndices(2));

		mbt = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 100, 1, 3, 0.2, 1342));
		n = empty.append(mbt).append(empty);
		addAll(tests, n.append(n, false), Util.genColsIndices(2));

		m = n.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), -1), null);
		addAll(tests, n, cols, cs);
		addAll(tests, m, cols, cs);
		addAll(tests, m.append(m, false), Util.genColsIndices(2));

	}

	protected static void addOverCharLong(ArrayList<Object[]> tests) {

		final int len = 300;
		int[] cols = new int[] {0};
		MatrixBlock mb;

		MatrixBlock e = new MatrixBlock(1, len / 2, true);
		MatrixBlock f = new MatrixBlock(1, len / 2 - 3, 1.0);

		addAll(tests, f.append(f).append(e).append(e).append(f), cols);

		mb = new MatrixBlock(2, 10 + len, 2.0).append(new MatrixBlock(2, len + 30, 0.0))
			.append(new MatrixBlock(2, 20 + 2 * len, 2.0));
		addAll(tests, mb, new int[] {0, 1});

	}


	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n\nColGroupBaseTest!\n");
		sb.append(base);
		sb.append("\n");
		sb.append(other);
		sb.append("\n");
		return sb.toString();
	}
}
