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

package org.apache.sysds.test.component.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockLDRB;
import org.apache.sysds.runtime.data.DenseBlockLFP64;
import org.apache.sysds.runtime.io.DeltaKernelUtils;
import org.apache.sysds.runtime.io.ReaderDelta;
import org.apache.sysds.runtime.io.ReaderDeltaParallel;
import org.apache.sysds.runtime.io.WriterDelta;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.Row;
import io.delta.kernel.internal.InternalScanFileUtils;
import io.delta.kernel.types.BinaryType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DateType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.types.TimestampType;

/**
 * Targeted tests for the error/defensive branches of the native Delta matrix
 * read/write code that the round-trip and interop tests do not reach: malformed
 * per-file statistics, unsupported column types, unsupported stream operations,
 * bad table paths, and the non-dense writer input path.
 *
 * <p>A few of these branches guard against inputs that the SystemDS writer and
 * the Delta Kernel scan API never produce in a normal round trip (e.g. a
 * statistics JSON without {@code numRecords}, or a column type code outside the
 * supported set). They are exercised here by mocking the Delta Kernel data
 * objects and invoking the (package-private) helpers reflectively, rather than
 * widening their production visibility purely for testing.
 */
public class DeltaMatrixCoverageTest {

	// ---------------------------------------------------------------------
	// public defensive paths
	// ---------------------------------------------------------------------

	@Test
	public void qualifyRejectsUnknownFilesystemScheme() {
		try {
			DeltaKernelUtils.qualify("nosuchfs://host/path/to/table");
			fail("expected a DMLRuntimeException for an unresolvable table path");
		}
		catch(DMLRuntimeException ex) {
			assertTrue("message should reference the bad path, got: " + ex.getMessage(),
				ex.getMessage() != null && ex.getMessage().contains("Delta table path"));
		}
	}

	@Test
	public void typeCodeReturnsNegativeForUnsupportedTypes() {
		//non-numeric / unsupported Delta types must map to the sentinel -1 so the
		//reader can reject them with a clear message rather than mis-decoding.
		assertEquals(-1, DeltaKernelUtils.typeCode(DateType.DATE));
		assertEquals(-1, DeltaKernelUtils.typeCode(TimestampType.TIMESTAMP));
		assertEquals(-1, DeltaKernelUtils.typeCode(BinaryType.BINARY));
	}

	@Test(expected = UnsupportedOperationException.class)
	public void readerRejectsInputStream() throws Exception {
		new ReaderDelta().readMatrixFromInputStream(null, 1, 1, -1, -1);
	}

	@Test(expected = UnsupportedOperationException.class)
	public void writerRejectsStreamWrite() throws Exception {
		new WriterDelta().writeMatrixFromStream("dummy", null, 1, 1, -1);
	}

	// ---------------------------------------------------------------------
	// mocked / reflective coverage of internal defensive branches
	// ---------------------------------------------------------------------

	@Test
	public void numRecordsHandlesAbsentNullAndMalformedStats() throws Exception {
		//no "stats" field at all -> -1
		assertEquals(-1, numRecords(addFileRow(new StructType().add("path", StringType.STRING), false, null)));
		//stats column present but null-at -> -1
		assertEquals(-1, numRecords(addFileRow(statsSchema(), true, null)));
		//stats string explicitly null -> -1
		assertEquals(-1, numRecords(addFileRow(statsSchema(), false, null)));
		//malformed JSON -> JsonProcessingException -> -1
		assertEquals(-1, numRecords(addFileRow(statsSchema(), false, "{not valid json")));
		//valid JSON but no numRecords field -> -1
		assertEquals(-1, numRecords(addFileRow(statsSchema(), false, "{\"minValues\":{}}")));
		//well-formed stats -> the parsed count
		assertEquals(1234L, numRecords(addFileRow(statsSchema(), false, "{\"numRecords\":1234}")));
	}

	@Test
	public void getDoubleValueRejectsUnknownTypeCode() throws Exception {
		Method m = ReaderDelta.class.getDeclaredMethod("getDoubleValue", ColumnVector.class, int.class, int.class);
		m.setAccessible(true);
		try {
			//type code outside the supported T_* set; the switch default must throw
			//before touching the (null) vector.
			m.invoke(null, (ColumnVector) null, 0, 999);
			fail("expected a DMLRuntimeException for an unsupported type code");
		}
		catch(InvocationTargetException ite) {
			assertTrue(ite.getCause() instanceof DMLRuntimeException);
		}
	}

	@Test
	public void numericTypeCodeRejectsNonNumericType() throws Exception {
		Method m = ReaderDelta.class.getDeclaredMethod("numericTypeCode", DataType.class, String.class);
		m.setAccessible(true);
		try {
			m.invoke(null, DateType.DATE, "d");
			fail("expected a DMLRuntimeException for a non-numeric column type");
		}
		catch(InvocationTargetException ite) {
			assertTrue(ite.getCause() instanceof DMLRuntimeException);
		}
	}

	@Test
	public void parallelReadWrapsFileFailure() throws Exception {
		//a per-file decode failure in the parallel reader must surface as a single
		//clear IOException (the awaitFileTasks catch), not a raw executor error.
		//Provoke it by deleting one data file after the table (and its log) exist.
		MatrixBlock in = TestUtils.generateTestMatrixBlock(100_000, 8, -10, 10, 1.0, 13);
		in.recomputeNonZeros();
		DMLConfig conf = new DMLConfig();
		conf.setTextValue(DMLConfig.DELTA_WRITER_TARGET_FILE_SIZE, String.valueOf(256L * 1024));
		ConfigurationManager.setLocalConfig(conf);
		Path dir = Files.createTempDirectory("sysds_delta_fail_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());

			//delete one parquet data file; the transaction log still references it,
			//so the scan enumerates it but the decode task fails.
			File victim;
			try( java.util.stream.Stream<Path> s = Files.walk(new File(tablePath).toPath()) ) {
				victim = s.filter(p -> p.toString().endsWith(".parquet"))
					.findFirst().map(Path::toFile).orElse(null);
			}
			assertTrue("expected at least one data file to delete", victim != null && victim.delete());

			try {
				new ReaderDeltaParallel().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
				fail("expected an IOException when a Delta data file is missing");
			}
			catch(java.io.IOException ex) {
				assertTrue("message should describe the failed parallel read, got: " + ex.getMessage(),
					ex.getMessage() != null && ex.getMessage().contains("parallel read of Delta table"));
			}
		}
		finally {
			ConfigurationManager.clearLocalConfigs();
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	// ---------------------------------------------------------------------
	// non-dense writer input path
	// ---------------------------------------------------------------------

	@Test
	public void sparseFormatMatrixRoundTrips() throws Exception {
		//a sparse-backed MatrixBlock takes the writer's non-contiguous path (no
		//direct double[] view), exercising MatrixColumnVector.get via MatrixBlock.
		MatrixBlock in = TestUtils.generateTestMatrixBlock(2000, 7, -5, 5, 0.05, 13);
		in.recomputeNonZeros();
		in.examSparsity();
		assertTrue("input should be in sparse format to exercise the non-dense path", in.isInSparseFormat());

		Path dir = Files.createTempDirectory("sysds_delta_sparse_");
		String tablePath = new File(dir.toFile(), "table").getAbsolutePath();
		try {
			new WriterDelta().writeMatrixToHDFS(in, tablePath,
				in.getNumRows(), in.getNumColumns(), -1, in.getNonZeros());
			MatrixBlock out = new ReaderDelta().readMatrixFromHDFS(tablePath, -1, -1, -1, -1);
			assertEquals("rows", in.getNumRows(), out.getNumRows());
			assertEquals("cols", in.getNumColumns(), out.getNumColumns());
			TestUtils.compareMatrices(in, out, 1e-12, "sparse-format-roundtrip");
		}
		finally {
			FileUtils.deleteQuietly(dir.toFile());
		}
	}

	@Test
	public void fillDenseHandlesNonContiguousBlock() throws Exception {
		//the dense fill normally hits the contiguous fast path; force a multi-block
		//(non-contiguous) dense block so the row-by-row fallback is exercised. Such
		//blocks only arise for matrices beyond a single contiguous array, so we
		//shrink the per-block allocation cap to provoke it on a tiny matrix.
		int rows = 5, cols = 4;
		int savedMaxAlloc = DenseBlockLDRB.MAX_ALLOC;
		DenseBlock db;
		try {
			DenseBlockLDRB.MAX_ALLOC = 2 * cols; //~2 rows per block -> multiple blocks
			db = new DenseBlockLFP64(new int[] {rows, cols});
		}
		finally {
			DenseBlockLDRB.MAX_ALLOC = savedMaxAlloc;
		}
		assertTrue("expected a non-contiguous (multi-block) dense block", !db.isContiguous());

		//two row-major batches (3 rows + 2 rows) covering all 5 rows
		double[] b0 = new double[3 * cols];
		double[] b1 = new double[2 * cols];
		for( int r = 0; r < 3; r++ )
			for( int c = 0; c < cols; c++ )
				b0[r * cols + c] = cell(r, c);
		for( int r = 0; r < 2; r++ )
			for( int c = 0; c < cols; c++ )
				b1[r * cols + c] = cell(3 + r, c);
		java.util.ArrayList<double[]> batches = new java.util.ArrayList<>();
		batches.add(b0);
		batches.add(b1);

		MatrixBlock ret = new MatrixBlock(rows, cols, db);
		Method m = ReaderDelta.class.getDeclaredMethod("fillDense", MatrixBlock.class, java.util.ArrayList.class);
		m.setAccessible(true);
		m.invoke(null, ret, batches);

		for( int r = 0; r < rows; r++ )
			for( int c = 0; c < cols; c++ )
				assertEquals("r" + r + " c" + c, cell(r, c), ret.getDenseBlock().get(r, c), 0.0);
	}

	private static double cell(int r, int c) {
		return r * 10 + c;
	}

	// ---------------------------------------------------------------------
	// helpers
	// ---------------------------------------------------------------------

	private static StructType statsSchema() {
		return new StructType().add("stats", StringType.STRING);
	}

	/**
	 * Build a mocked scan-file row whose AddFile child has the given schema, null
	 * flag and (when not null) stats string, matching what {@code numRecords} reads.
	 */
	private static Row addFileRow(StructType addSchema, boolean statsNull, String statsValue) {
		Row outer = mock(Row.class);
		Row add = mock(Row.class);
		when(outer.getStruct(InternalScanFileUtils.ADD_FILE_ORDINAL)).thenReturn(add);
		when(add.getSchema()).thenReturn(addSchema);
		int statsOrd = addSchema.fieldNames().indexOf("stats");
		if( statsOrd >= 0 ) {
			when(add.isNullAt(statsOrd)).thenReturn(statsNull);
			if( !statsNull )
				when(add.getString(statsOrd)).thenReturn(statsValue);
		}
		return outer; //the scan-file row numRecords consumes (its AddFile child is 'add')
	}

	private static long numRecords(Row scanFileRow) throws Exception {
		Method m = DeltaKernelUtils.class.getDeclaredMethod("numRecords", Row.class);
		m.setAccessible(true);
		return (Long) m.invoke(null, scanFileRow);
	}
}
