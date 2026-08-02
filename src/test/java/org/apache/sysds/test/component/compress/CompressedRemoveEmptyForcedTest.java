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

import java.util.EnumSet;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Verifies that {@code removeEmptyOperations} with a selection vector degrades gracefully (decompresses) for
 * column-group encodings that do not implement index-only row/column removal (e.g. OLE/RLE), rather than throwing
 * {@link org.apache.commons.lang3.NotImplementedException}.
 */
public class CompressedRemoveEmptyForcedTest {

	private static final int ROWS = 500;
	private static final int COLS = 4;

	@Test
	public void removeEmptyRowsFallbackOLE() {
		runFallback(CompressionType.OLE, true);
	}

	@Test
	public void removeEmptyRowsFallbackRLE() {
		runFallback(CompressionType.RLE, true);
	}

	@Test
	public void removeEmptyColsFallbackRLE() {
		runFallback(CompressionType.RLE, false);
	}

	private void runFallback(CompressionType ct, boolean rows) {
		MatrixBlock mb = CompressibleInputGenerator.getInput(ROWS, COLS, ct, 10, 0.6, 7);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0)
			.setValidCompressions(EnumSet.of(ct));
		MatrixBlock compressed = CompressedMatrixBlockFactory.compress(mb, 1, csb).getLeft();
		assertTrue("Expected the input to compress into a " + ct + " backed block",
			compressed instanceof CompressedMatrixBlock);
		CompressedMatrixBlock cmb = (CompressedMatrixBlock) compressed;
		assertTrue("Expected at least one " + ct + " column group to exercise the fallback path",
			containsType(cmb, ct));

		// Use a strict subset selection so the column path reaches removeEmptyColsSubset (which throws
		// NotImplementedException for OLE/RLE) rather than the copyAndSet all-selected shortcut.
		final MatrixBlock select;
		if(rows) {
			select = new MatrixBlock(ROWS, 1, false);
			for(int i = 0; i < ROWS; i += 2)
				select.set(i, 0, 1);
		}
		else {
			select = new MatrixBlock(1, COLS, false);
			select.set(0, 0, 1);
		}

		// Must not throw NotImplementedException; must match the uncompressed reference via decompression fallback.
		MatrixBlock actual = cmb.removeEmptyOperations(null, rows, false, select);
		MatrixBlock expected = mb.removeEmptyOperations(null, rows, false, select);
		TestUtils.compareMatrices(expected, actual, 0.0, "removeEmpty fallback for " + ct + " rows=" + rows);
	}

	private static boolean containsType(CompressedMatrixBlock cmb, CompressionType ct) {
		for(AColGroup g : cmb.getColGroups())
			if(g.getCompType() == ct)
				return true;
		return false;
	}
}
