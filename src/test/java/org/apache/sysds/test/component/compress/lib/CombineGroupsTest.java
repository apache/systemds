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

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class CombineGroupsTest {

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
			MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 1, 1, 1, 0.5, 230);
			CompressedMatrixBlock ac = com(a);
			MatrixBlock b = TestUtils.generateTestMatrixBlock(100, 1, 1, 1, 0.5, 132);
			CompressedMatrixBlock bc = com(b);
			CompressedMatrixBlock buc = ucom(b); // uncompressed col group
			MatrixBlock c = new MatrixBlock(100, 1, 1.34);
			CompressedMatrixBlock cc = com(c); // const
			MatrixBlock e = new MatrixBlock(100, 1, 0);
			CompressedMatrixBlock ec = com(e); // empty

			// Default DDC case
			tests.add(new Object[] {a, b, ac, bc});

			// Empty and Const cases.
			tests.add(new Object[] {a, c, ac, cc});
			tests.add(new Object[] {a, e, ac, ec});
			tests.add(new Object[] {c, a, cc, ac});
			tests.add(new Object[] {e, a, ec, ac});
			tests.add(new Object[] {e, e, ec, ec});
			tests.add(new Object[] {c, c, cc, cc});
			tests.add(new Object[] {c, e, cc, ec});
			tests.add(new Object[] {e, c, ec, cc});

			// Uncompressed Case
			tests.add(new Object[] {a, b, ac, buc});
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
			CompressedMatrixBlock cc = appendNoMerge();
		

			TestUtils.compareMatricesBitAvgDistance(c, cc, 0, 0, "Not the same verification");
			CompressedMatrixBlock ccc = (CompressedMatrixBlock) cc;
			List<AColGroup> groups = ccc.getColGroups();
			if(groups.size() > 1){

				AColGroup cg = CLALibCombineGroups.combine(groups.get(0), groups.get(1));
				ccc.allocateColGroup(cg);
				TestUtils.compareMatricesBitAvgDistance(c, ccc, 0, 0, "Not the same combined");
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}


	private CompressedMatrixBlock appendNoMerge(){
		CompressedMatrixBlock cc = new CompressedMatrixBlock(ac.getNumRows(), ac.getNumColumns() + bc.getNumColumns());
		appendColGroups(cc, ac.getColGroups(), bc.getColGroups(), ac.getNumColumns());
		cc.setNonZeros(ac.getNonZeros() + bc.getNonZeros());
		cc.setOverlapping(ac.isOverlapping() || bc.isOverlapping());
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
}
