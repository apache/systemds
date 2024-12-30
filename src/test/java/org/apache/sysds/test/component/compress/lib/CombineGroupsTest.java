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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CombineGroupsTest {
	protected static final Log LOG = LogFactory.getLog(CombineGroupsTest.class.getName());

	final MatrixBlock a;
	final MatrixBlock b;
	final CompressedMatrixBlock ac;
	final CompressedMatrixBlock bc;

	public CombineGroupsTest(MatrixBlock a, MatrixBlock b, CompressedMatrixBlock ac, CompressedMatrixBlock bc) {
		this.a = a;
		this.b = b;
		this.ac = ac;
		this.bc = bc;
	}

	@Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();

		try {
			int[] nCols = new int[] {1, 2, 5};
			for(int nCol : nCols) {

				MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 100, 103, 0.5, 230);
				a = TestUtils.ceil(a);
				a = cbindN(a, nCol);
				CompressedMatrixBlock ac = com(a);
				MatrixBlock b = TestUtils.generateTestMatrixBlock(100, 1, 13, 15, 0.5, 132);
				b = TestUtils.ceil(b);
				b = cbindN(b, nCol);
				CompressedMatrixBlock bc = com(b);
				CompressedMatrixBlock buc = ucom(b); // uncompressed col group
				MatrixBlock c = new MatrixBlock(100, nCol, 1.34);
				CompressedMatrixBlock cc = com(c); // const
				MatrixBlock e = new MatrixBlock(100, nCol, 0);
				CompressedMatrixBlock ec = com(e); // empty

				MatrixBlock u = TestUtils.generateTestMatrixBlock(100, 1, 0, 1, 1.0, 2315);
				u = cbindN(u, nCol);
				CompressedMatrixBlock uuc = ucom(u);

				// Default DDC case
				tests.add(new Object[] {a, b, ac, bc});

				// Empty and Const cases.
				tests.add(new Object[] {a, c, ac, cc}); // const ddc
				tests.add(new Object[] {c, a, cc, ac});
				tests.add(new Object[] {a, e, ac, ec}); // empty ddc
				tests.add(new Object[] {e, a, ec, ac});
				tests.add(new Object[] {c, e, cc, ec}); // empty const
				tests.add(new Object[] {e, c, ec, cc});
				tests.add(new Object[] {e, e, ec, ec}); // empty empty
				tests.add(new Object[] {c, c, cc, cc}); // const const

				// Uncompressed Case
				tests.add(new Object[] {a, b, ac, buc}); // compressable uncompressed group
				tests.add(new Object[] {b, a, buc, ac});
				tests.add(new Object[] {a, u, ac, uuc}); // incompressable input
				tests.add(new Object[] {u, a, uuc, ac});
				tests.add(new Object[] {u, u, uuc, uuc}); // both sides incompressable

				MatrixBlock s = TestUtils.generateTestMatrixBlock(100, 1, 1, 3, 0.10, 123);
				s = TestUtils.ceil(s);
				s = cbindN(s, nCol);
				CompressedMatrixBlock sc = com(s); // SDCZeroSingle

				MatrixBlock s2 = TestUtils.generateTestMatrixBlock(100, 1, 0, 3, 0.2, 321);
				s2 = TestUtils.ceil(s2);
				s2 = cbindN(s2, nCol);
				CompressedMatrixBlock s2c = com(s2); // SDCZero

				// SDC cases
				tests.add(new Object[] {s, a, sc, ac});
				tests.add(new Object[] {a, s, ac, sc});
				tests.add(new Object[] {s2, a, s2c, ac});
				tests.add(new Object[] {a, s2, ac, s2c});

				tests.add(new Object[] {s, s, sc, sc});
				tests.add(new Object[] {s, s2, sc, s2c});

				// empty and const SDC
				tests.add(new Object[] {s, e, sc, ec});
				tests.add(new Object[] {s, c, sc, cc});
				tests.add(new Object[] {e, s, ec, sc});
				tests.add(new Object[] {c, s, cc, sc});
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	private static CompressedMatrixBlock com(MatrixBlock m) {
		return (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(m).getLeft();
	}

	private static CompressedMatrixBlock ucom(MatrixBlock m) {
		return CompressedMatrixBlockFactory.genUncompressedCompressedMatrixBlock(m);
	}

	@Test
	public void combineTest() {
		try {

			// combined.
			MatrixBlock c = a.append(b);
			CompressedMatrixBlock cc = appendNoMerge(ac, bc);

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(0), groups.get(1), c.getNumRows());
				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}

		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void combineWithExtraEmptyColumnsBefore() {
		try {
			MatrixBlock e = new MatrixBlock(a.getNumRows(), 2, true);
			// combined.
			MatrixBlock c = e.append(a).append(b);
			CompressedMatrixBlock ec = CompressedMatrixBlockFactory.createConstant(a.getNumRows(), 2, 0.0);
			CompressedMatrixBlock cc = appendNoMerge(ec, appendNoMerge(ac, bc));

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(1), groups.get(2), c.getNumRows());
				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}

		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineWithExtraConstColumnsBefore() {
		try {
			MatrixBlock e = new MatrixBlock(a.getNumRows(), 2, 1.0);
			// combined.
			MatrixBlock c = e.append(a).append(b);
			CompressedMatrixBlock ec = CompressedMatrixBlockFactory.createConstant(a.getNumRows(), 2, 1.0);
			CompressedMatrixBlock cc = appendNoMerge(ec, appendNoMerge(ac, bc));

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(1), groups.get(2), c.getNumRows());
				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}

		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineWithExtraConstColumnsBetween() {
		try {
			MatrixBlock e = new MatrixBlock(a.getNumRows(), 2, 1.0);
			// combined.
			MatrixBlock c = a.append(e).append(b);
			CompressedMatrixBlock ec = CompressedMatrixBlockFactory.createConstant(a.getNumRows(), 2, 1.0);
			CompressedMatrixBlock cc = appendNoMerge(ac, appendNoMerge(ec, bc));

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(0), groups.get(2), c.getNumRows());
				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineWithExtraConstColumnsAfter() {
		try {
			MatrixBlock e = new MatrixBlock(a.getNumRows(), 2, 1.0);
			// combined.
			MatrixBlock c = a.append(b).append(e);
			CompressedMatrixBlock ec = CompressedMatrixBlockFactory.createConstant(a.getNumRows(), 2, 1.0);
			CompressedMatrixBlock cc = appendNoMerge(ac, appendNoMerge(bc, ec));

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(0), groups.get(1), c.getNumRows());
				ccc.allocateColGroup(cg);

				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}

		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void combineMixingColumnIndexes() {

		try {
			if(a.getNumColumns() + b.getNumColumns() == 2)
				return; // not relevant test
			// combined.
			MatrixBlock c = a.append(b);
			CompressedMatrixBlock cc = appendNoMerge(ac, bc);

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			// mix columns...
			int[] mix = new int[c.getNumColumns()];
			inc(mix);
			shuffle(mix, 13);
			c = applyShuffle(c, mix);
			cc = applyCompressedShuffle(cc, mix);
			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same after shuffle verification: ");

			CompressedMatrixBlock ccc = cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1) {

				AColGroup cg = CLALibCombineGroups.combine(groups.get(0), groups.get(1), c.getNumRows());
				assertTrue(cg.getColIndices().isSorted());

				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined ");
			}

		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("Not allowed")) {

				e.printStackTrace();
				fail(e.getMessage());
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static CompressedMatrixBlock appendNoMerge(CompressedMatrixBlock a, CompressedMatrixBlock b) {
		CompressedMatrixBlock cc = new CompressedMatrixBlock(a.getNumRows(), a.getNumColumns() + b.getNumColumns());
		appendColGroups(cc, a.getColGroups(), b.getColGroups(), a.getNumColumns());
		cc.setNonZeros(a.getNonZeros() + b.getNonZeros());
		cc.setOverlapping(a.isOverlapping() || b.isOverlapping());
		return cc;
	}

	private static void appendColGroups(CompressedMatrixBlock ret, List<AColGroup> left, List<AColGroup> right,
		int leftNumCols) {

		// shallow copy of lhs column groups
		ret.allocateColGroupList(new ArrayList<AColGroup>(left.size() + right.size()));

		for(AColGroup group : left)
			ret.getColGroups().add(group);

		for(AColGroup group : right)
			ret.getColGroups().add(group.shiftColIndices(leftNumCols));

	}

	private static MatrixBlock cbindN(MatrixBlock a, int n) {
		MatrixBlock b = a;
		for(int i = 1; i < n; i++) {
			b = b.append(a);
		}
		return b;
	}

	private static void inc(int[] ar) {
		for(int i = 0; i < ar.length; i++) {
			ar[i] = i;
		}
	}

	private static void shuffle(int[] ar, int seed) {
		Random r = new Random();
		for(int i = 0; i < ar.length; i++) {
			int f = r.nextInt(ar.length);
			int t = r.nextInt(ar.length);
			int tmp = ar[t];
			ar[t] = ar[f];
			ar[f] = tmp;
		}
	}

	private static MatrixBlock applyShuffle(MatrixBlock a, int[] mix) {
		MatrixBlock ret = new MatrixBlock(a.getNumRows(), a.getNumColumns(), a.getNonZeros());
		for(int r = 0; r < a.getNumRows(); r++)
			for(int c = 0; c < a.getNumColumns(); c++)
				ret.set(r, mix[c], a.get(r, c));
		return ret;
	}

	private static CompressedMatrixBlock applyCompressedShuffle(CompressedMatrixBlock a, int[] mix) {
		List<AColGroup> in = a.getColGroups();
		List<AColGroup> out = new ArrayList<>();
		for(AColGroup g : in)
			out.add(moveCols(g, mix));
		a.allocateColGroupList(out);
		a.clearSoftReferenceToDecompressed();
		return a;
	}

	private static AColGroup moveCols(AColGroup g, int[] mix) {
		IColIndex gi = g.getColIndices();
		int[] newIndexes = new int[gi.size()];

		IIterate it = gi.iterator();
		while(it.hasNext()) {
			newIndexes[it.i()] = mix[it.v()];
			it.next();
		}

		g = g.copyAndSet(new ArrayIndex(newIndexes));
		g = g.sortColumnIndexes();
		return g;
	}
}
