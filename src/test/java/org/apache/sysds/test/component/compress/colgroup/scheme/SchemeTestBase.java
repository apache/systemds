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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.readers.ReadersTestCompareReaders;
import org.junit.Test;

public abstract class SchemeTestBase {
	protected static final Log LOG = LogFactory.getLog(SchemeTestBase.class.getName());

	static {
		CompressedMatrixBlock.debug = true;
	}

	MatrixBlock src;
	int distinct;
	ICLAScheme sh;

	@Test
	public void testEncode() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(20, src.getNumColumns(), 0, distinct, 0.9, 7));
			AColGroup out = sh.encode(in);
			MatrixBlock d = new MatrixBlock(in.getNumRows(), in.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(in, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testEncodeT() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 20, 0, distinct, 0.9, 7));
			AColGroup out = sh.encodeT(in);
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(in, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testEncode_sparse() {
		try {
			MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, 100, 0, distinct, 0.05, 7));
			AColGroup out = sh.encode(in);
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testEncode_sparseT() {
		try {

			MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, 100, 0, distinct, 0.05, 7));
			AColGroup out = sh.encodeT(in);
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumRows() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdate() {
		try {

			double newVal = distinct + 1;
			MatrixBlock in = new MatrixBlock(5, src.getNumColumns(), newVal);
			try {
				sh.encode(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
			}
			ICLAScheme shc = sh.clone();
			shc = shc.update(in);
			AColGroup out = shc.encode(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdateT() {
		try {

			double newVal = distinct + 1;
			MatrixBlock in = new MatrixBlock(src.getNumColumns(), 5, newVal);
			try {
				sh.encodeT(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
				// but we can also not have an exception thrown...
			}
			ICLAScheme shc = sh.clone();

			shc = shc.updateT(in);

			AColGroup out = shc.encodeT(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
			assertEquals(inSlice.getNumRows(), d.getNumColumns());
			assertEquals(inSlice.getNumColumns(), d.getNumRows());
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdateSparse() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(130, src.getNumColumns() + 30, 0, distinct + 1, 0.1, 7));
			if(!in.isInSparseFormat())
				throw new RuntimeException();
			try {
				sh.encode(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
			}
			ICLAScheme shc = sh.clone();
			shc = shc.update(in);
			AColGroup out = shc.encode(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdateSparseT() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 1000, 0, distinct + 1, 0.1, 7));
			if(!in.isInSparseFormat())
				throw new RuntimeException();
			try {
				sh.encodeT(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
				// but we can also not have an exception thrown...
			}
			ICLAScheme shc = sh.clone();
			shc = shc.updateT(in);

			AColGroup out = shc.encodeT(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index"))
				return; // all good
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testUpdateSparseTEmptyColumn() {
		try {

			MatrixBlock in = new MatrixBlock(src.getNumColumns(), 100, 0.0);
			MatrixBlock b = new MatrixBlock(1, 100, 1.0);
			in = in.append(b, false);
			in.denseToSparse(true);
			if(!in.isInSparseFormat())
				throw new RuntimeException();
			try {
				sh.encodeT(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
				// but we can also not have an exception thrown...
			}
			ICLAScheme shc = sh.clone();
			shc = shc.updateT(in);

			AColGroup out = shc.encodeT(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return; // all good expected exception
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testUpdateLargeBlock() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(130, src.getNumColumns(), 0, distinct + 3, 1.0, 7));
			in = ReadersTestCompareReaders.createMock(in);
			try {
				sh.encode(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
			}
			ICLAScheme shc = sh.clone();
			shc = shc.update(in);
			AColGroup out = shc.encode(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}

	}

	@Test
	public void testUpdateLargeBlockT() {
		try {

			MatrixBlock in = TestUtils
				.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 130, 0, distinct + 5, 1.0, 7));
			in = ReadersTestCompareReaders.createMock(in);
			try {
				sh.encodeT(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
				// but we can also not have an exception thrown...
			}
			ICLAScheme shc = sh.clone();

			shc = shc.updateT(in);

			AColGroup out = shc.encodeT(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdateEmpty() {
		try {

			MatrixBlock in = new MatrixBlock(5, src.getNumColumns(), 0.0);

			try {
				sh.encode(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
			}
			ICLAScheme shc = sh.clone();
			shc = shc.update(in);
			AColGroup out = shc.encode(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testUpdateEmptyT() {
		// 5 rows to encode transposed

		MatrixBlock in = new MatrixBlock(src.getNumColumns(), 5, 0.0);
		try {
			sh.encodeT(in);
		}
		catch(NullPointerException e) {
			// all good expected
			// we want to have an exception thrown if we try to encode something that is not possible to encode.
			// but we can also not have an exception thrown...
		}
		ICLAScheme shc = sh.clone();

		AColGroup out = shc.encodeT(in); // should be possible now.

		// now we learned how to encode. lets decompress the encoded.

		MatrixBlock d = new MatrixBlock( in.getNumColumns(), in.getNumRows(), false);
		d.allocateBlock();
		out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
		MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, in.getNumColumns() - 1);
		d.recomputeNonZeros();
		TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
	}

	@Test
	public void testUpdateEmptyMyCols() {
		try {

			MatrixBlock in = new MatrixBlock(5, src.getNumColumns(), 0.0);
			in = in.append(new MatrixBlock(5, 1, 1.0));

			try {
				sh.encode(in);
			}
			catch(NullPointerException e) {
				// all good expected
				// we want to have an exception thrown if we try to encode something that is not possible to encode.
			}
			ICLAScheme shc = sh.clone();
			shc = shc.update(in);
			AColGroup out = shc.encode(in); // should be possible now.
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}

	}

	@Test
	public void testUpdateEmptyMyColsT() {
		MatrixBlock in = new MatrixBlock(src.getNumColumns(), 5, 0.0);
		in = in.append(new MatrixBlock(src.getNumColumns(), 1, 1.0), true);
	
		try {
			sh.encodeT(in);
		}
		catch(NullPointerException e) {
			// all good expected
			// we want to have an exception thrown if we try to encode something that is not possible to encode.
			// but we can also not have an exception thrown...
		}
		ICLAScheme shc = sh.clone();

		shc = shc.updateT(in);

		AColGroup out = shc.encodeT(in); // should be possible now.
		// MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
		// d.allocateBlock();
		// out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
		// MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
		// d.recomputeNonZeros();
		// TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);

		MatrixBlock d = new MatrixBlock( in.getNumColumns(), in.getNumRows(), false);
		d.allocateBlock();
		out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
		MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, in.getNumColumns() - 1);
		d.recomputeNonZeros();
		TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
	}

	@Test
	public void testUpdateAndEncode() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns(), 0, newVal, 1.0, 7));
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeT() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 1.0, 7));
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeSparse() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils
			.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns() + 100, 0, newVal, 0.1, 7));
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeSparseT() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 0.1, 7));
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeSparseTEmptyColumn() {
		MatrixBlock in = new MatrixBlock(src.getNumColumns(), 10, 0.0);
		MatrixBlock b = new MatrixBlock(1, 10, 1.0);
		in = in.append(b, false);
		in.denseToSparse(true);
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeLarge() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns(), 0, newVal, 1.0, 7));

		in = ReadersTestCompareReaders.createMock(in);
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeLargeT() {
		double newVal = distinct + 4;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 1.0, 7));
		in = ReadersTestCompareReaders.createMock(in);
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns(), 0, newVal, 1.0, 7));
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeTManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 1.0, 7));
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeSparseManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils
			.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns() + 100, 0, newVal, 0.1, 7));
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeSparseTManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 0.1, 7));
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeLargeManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(100, src.getNumColumns(), 0, newVal, 1.0, 7));

		in = ReadersTestCompareReaders.createMock(in);
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeLargeTManyNew() {
		double newVal = distinct + 300;
		MatrixBlock in = TestUtils.round(TestUtils.generateTestMatrixBlock(src.getNumColumns(), 100, 0, newVal, 1.0, 7));
		in = ReadersTestCompareReaders.createMock(in);
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeEmpty() {
		MatrixBlock in = new MatrixBlock(100, src.getNumColumns(), 0);
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeEmptyT() {
		MatrixBlock in = new MatrixBlock(src.getNumColumns(), 100, 0);
		testUpdateAndEncodeT(in);
	}

	@Test
	public void testUpdateAndEncodeEmptyInCols() {
		MatrixBlock in = new MatrixBlock(100, src.getNumColumns(), 0.0);
		in = in.append(new MatrixBlock(100, src.getNumColumns(), 1.0));
		testUpdateAndEncode(in);
	}

	@Test
	public void testUpdateAndEncodeEmptyInColsT() {
		MatrixBlock in = new MatrixBlock(src.getNumColumns(), 100, 0.0);
		in = in.append(new MatrixBlock(src.getNumColumns(), 100, 1.0), false);
		testUpdateAndEncodeT(in);
	}

	public void testUpdateAndEncode(MatrixBlock in) {
		try {

			Pair<ICLAScheme, AColGroup> r = sh.clone().updateAndEncode(in);
			AColGroup out = r.getValue();
			MatrixBlock d = new MatrixBlock(in.getNumRows(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumRows());
			MatrixBlock inSlice = in.slice(0, in.getNumRows() - 1, 0, src.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, d, 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	public void testUpdateAndEncodeT(MatrixBlock in) {
		try {
			Pair<ICLAScheme, AColGroup> r = sh.clone().updateAndEncodeT(in);
			AColGroup out = r.getValue();
			MatrixBlock d = new MatrixBlock(in.getNumColumns(), src.getNumColumns(), false);
			d.allocateBlock();
			out.decompressToDenseBlock(d.getDenseBlock(), 0, in.getNumColumns());
			MatrixBlock inSlice = in.slice(0, src.getNumColumns() - 1, 0, in.getNumColumns() - 1);
			d.recomputeNonZeros();
			TestUtils.compareMatricesBitAvgDistance(inSlice, LibMatrixReorg.transpose(d), 0, 0);
		}
		catch(Exception e) {
			if(e.getMessage().contains("Invalid SDC group that contains index with size == numRows"))
				return;// all good
			e.printStackTrace();
			fail(e.getMessage() + " " + sh);
		}
	}

	@Test
	public void testToString() {
		sh.toString();
	}

}
