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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDeltaDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class ColGroupFactoryDeltaDDCTest {

	@Test
	public void testCompressDeltaDDCSingleColumnWithGaps() {
		MatrixBlock mb = new MatrixBlock(10, 1, true);
		mb.set(0, 0, 10);
		mb.set(5, 0, 15);
		mb.set(9, 0, 20);

		IColIndex cols = ColIndexFactory.create(1);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 3;
		final EstimationFactors f = new EstimationFactors(3, nRow, offs, 0.3);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be DeltaDDC", groups.get(0) instanceof ColGroupDeltaDDC);
	}

	@Test
	public void testCompressDeltaDDCSingleColumnEmpty() {
		MatrixBlock mb = new MatrixBlock(10, 1, true);

		IColIndex cols = ColIndexFactory.create(1);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 0;
		final EstimationFactors f = new EstimationFactors(0, nRow, offs, 0.0);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be Empty", groups.get(0) instanceof ColGroupEmpty);
	}

	@Test
	public void testCompressDeltaDDCMultiColumnWithGaps() {
		MatrixBlock mb = new MatrixBlock(20, 2, true);
		mb.set(0, 0, 10);
		mb.set(0, 1, 20);
		mb.set(5, 0, 15);
		mb.set(5, 1, 25);
		mb.set(10, 0, 20);
		mb.set(10, 1, 30);
		mb.set(15, 0, 25);
		mb.set(15, 1, 35);

		IColIndex cols = ColIndexFactory.create(2);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 4;
		final EstimationFactors f = new EstimationFactors(4, nRow, offs, 0.2);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be DeltaDDC", groups.get(0) instanceof ColGroupDeltaDDC);
	}

	@Test
	public void testCompressDeltaDDCMultiColumnEmpty() {
		MatrixBlock mb = new MatrixBlock(10, 2, true);

		IColIndex cols = ColIndexFactory.create(2);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 0;
		final EstimationFactors f = new EstimationFactors(0, nRow, offs, 0.0);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be Empty", groups.get(0) instanceof ColGroupEmpty);
	}

	@Test
	public void testCompressDeltaDDCMultiColumnSparseWithGaps() {
		MatrixBlock mb = new MatrixBlock(50, 3, true);
		mb.set(0, 0, 1);
		mb.set(0, 1, 2);
		mb.set(0, 2, 3);
		mb.set(10, 0, 11);
		mb.set(10, 1, 12);
		mb.set(10, 2, 13);
		mb.set(20, 0, 21);
		mb.set(20, 1, 22);
		mb.set(20, 2, 23);
		mb.set(30, 0, 31);
		mb.set(30, 1, 32);
		mb.set(30, 2, 33);
		mb.set(40, 0, 41);
		mb.set(40, 1, 42);
		mb.set(40, 2, 43);

		IColIndex cols = ColIndexFactory.create(3);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 5;
		final EstimationFactors f = new EstimationFactors(5, nRow, offs, 0.1);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be DeltaDDC", groups.get(0) instanceof ColGroupDeltaDDC);
	}

	@Test
	public void testCompressDeltaDDCSingleColumnDense() {
		MatrixBlock mb = new MatrixBlock(10, 1, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 10; i++) {
			mb.set(i, 0, i + 1);
		}

		IColIndex cols = ColIndexFactory.create(1);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 10;
		final EstimationFactors f = new EstimationFactors(10, nRow, offs, 1.0);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be DeltaDDC", groups.get(0) instanceof ColGroupDeltaDDC);
	}

	@Test
	public void testCompressDeltaDDCMultiColumnDense() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 10; i++) {
			mb.set(i, 0, i + 1);
			mb.set(i, 1, (i + 1) * 2);
		}

		IColIndex cols = ColIndexFactory.create(2);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder();
		CompressionSettings cs = csb.create();

		final int nRow = mb.getNumRows();
		final int offs = 10;
		final EstimationFactors f = new EstimationFactors(10, nRow, offs, 1.0);
		final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
		es.add(new CompressedSizeInfoColGroup(cols, f, 314152, CompressionType.DeltaDDC));
		final CompressedSizeInfo csi = new CompressedSizeInfo(es);

		List<AColGroup> groups = ColGroupFactory.compressColGroups(mb, csi, cs);
		assertNotNull("Compression should succeed", groups);
		assertEquals("Should have one column group", 1, groups.size());
		assertTrue("Should be DeltaDDC", groups.get(0) instanceof ColGroupDeltaDDC);
	}

}

