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

package org.apache.sysds.test.component.compress.io;

import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.io.CompressedWriteBlock;
import org.apache.sysds.runtime.compress.io.ReaderCompressed;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class IONegativeTest {
	@Test(expected = RuntimeException.class)
	public void negativeBLen() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, null, 0, 0, -1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void zeroBLen() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, null, 0, 0, 0, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void diagonal() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, null, 0, 0, 1, 0, true);
	}

	@Test(expected = RuntimeException.class)
	public void noPath() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, null, 0, 0, 1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void noMatrix() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, "some_path", 0, 0, 1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void matrixRowsColsNotSame() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(new MatrixBlock(10, 10, 0.0), "some_path", 0, 0, 1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void matrixColsNotSameAsCLen() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(new MatrixBlock(10, 10, 0.0), "some_path", 10, 0, 1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void matrixRowsNotSameAsRLen() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(new MatrixBlock(10, 10, 0.0), "some_path", 0, 10, 1, 0, false);
	}

	@Test(expected = RuntimeException.class)
	public void writeEmptyZeroNRow() throws IOException {
		writeEmpty(null, 0, 100, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeEmptyNegativeNRow() throws IOException {
		writeEmpty(null, -13, 100, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeLargeNRow() throws IOException {
		writeEmpty(null, (long) Integer.MAX_VALUE + 1L, 100, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeEmptyZeroNCow() throws IOException {
		writeEmpty(null, 100, 0, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeEmptyNegativeNCow() throws IOException {
		writeEmpty(null, 100, -3241, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeLargeNCol() throws IOException {
		writeEmpty(null, 100, (long) Integer.MAX_VALUE + 1L, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeNullPath() throws IOException {
		writeEmpty(null, 100, 100, 100);
	}

	@Test(expected = RuntimeException.class)
	public void writeZeroBlock() throws IOException {
		writeEmpty(null, 100, 100, 0);
	}

	@Test(expected = RuntimeException.class)
	public void writeNegativeBlock() throws IOException {
		writeEmpty(null, 100, 100, -1);
	}

	private static void writeEmpty(String path, long nRows, long nCols, int blen) throws IOException {
		WriterCompressed.create(null).writeEmptyMatrixToHDFS(path, nRows, nCols, blen);
	}

	@Test(expected = NotImplementedException.class)
	public void readFromStream() throws IOException {
		ReaderCompressed r = ReaderCompressed.create();
		r.readMatrixFromInputStream(null, 0, 0, 0, 0);
	}

	@Test(expected = RuntimeException.class)
	public void compareToWriteBlock() {
		CompressedWriteBlock a = new CompressedWriteBlock();
		CompressedWriteBlock b = new CompressedWriteBlock();
		a.compareTo(b);
	}

	@Test(expected = DMLRuntimeException.class)
	public void writeDiagonalCompressedInterface2() throws IOException {
		WriterCompressed.writeCompressedMatrixToHDFS(null, "ada", 333L, 222L, 24, 130L, true);
	}
}
