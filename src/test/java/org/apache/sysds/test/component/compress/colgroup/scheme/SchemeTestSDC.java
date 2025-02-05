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

package org.apache.sysds.test.component.compress.colgroup.scheme;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.ComEstExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class SchemeTestSDC extends SchemeTestBase {
	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		try {
			tests.add(new Object[] {TestUtils.round(TestUtils.generateTestMatrixBlock(1023, 3, 0, 3, 0.7, 7)), 3});
			tests.add(new Object[] {TestUtils.round(TestUtils.generateTestMatrixBlock(1023, 1, 0, 10, 0.7, 7)), 10});
			tests.add(new Object[] {TestUtils.generateTestMatrixBlock(1023, 1, 1, 1, 0.7, 7), 1});
			tests.add(new Object[] {new MatrixBlock(100, 1, 0), 0});
			tests.add(new Object[] {new MatrixBlock(100, 3, 0), 0});
			tests.add(new Object[] {new MatrixBlock(100, 50, 0), 0});
			MatrixBlock t = new MatrixBlock(100, 3, 1.0);
			MatrixBlock a = TestUtils.generateTestMatrixBlock(100, 3, 1,1,0.5,4);
			t = t.append(a, false);
			
			tests.add(new Object[]{t, 1});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	public SchemeTestSDC(MatrixBlock src, int distinct) {
		try {
			this.src = src;
			this.distinct = distinct;
			IColIndex colIndexes = ColIndexFactory.create(src.getNumColumns());
			CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(CompressionType.SDC)).create();
			final CompressedSizeInfoColGroup cgi = new ComEstExact(src, cs).getColGroupInfo(colIndexes);

			final CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			final List<AColGroup> groups = ColGroupFactory.compressColGroups(src, csi, cs, 1);
			AColGroup g = groups.get(0);
			sh = g.getCompressionScheme();
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
