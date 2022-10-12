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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Test;

public class IOTest {

	protected static final Log LOG = LogFactory.getLog(IOTest.class.getName());

	final static String nameBeginning = "src/test/java/org/apache/sysds/test/component/compress/io/files"
		+ IOTest.class.getSimpleName() + "/";

	public IOTest() {
		synchronized(IOCompressionTestUtils.lock) {
			new File(nameBeginning).mkdirs();
		}
	}

	@AfterClass
	public static void cleanup() {
		IOCompressionTestUtils.deleteDirectory(new File(nameBeginning));
	}

	public static String getName() {
		return IOCompressionTestUtils.getName(nameBeginning);
	}

	@Test
	public void testWrite() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514));
		String n = getName();
		write(mb, n);
		File f = new File(n);
		assertTrue(f.isFile());
	}

	@Test
	public void testWriteAlreadyCompressed() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514));
		MatrixBlock mb2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
		writeAndRead(mb2);
	}

	@Test
	public void testWriteAndRead() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514)));
	}

	@Test
	public void testWriteNotCompressable() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(3, 3, 1, 3, 1.0, 2514)));
	}

	@Test
	public void testWriteNotCompressableV2() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(30, 3, 1, 10, 1.0, 2514)));
	}

	@Test
	public void testWriteNotCompressableV3() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(300, 3, 1, 50, 1.0, 2514)));
	}

	@Test
	public void testWriteNotCompressableRandomSparse() throws Exception {
		writeAndRead(TestUtils.generateTestMatrixBlock(300, 3, 1, 50, 0.1, 2514));
	}

	protected static void writeAndRead(MatrixBlock mb) throws Exception {
		String filename = getName();
		WriterCompressed.writeCompressedMatrixToHDFS(mb, filename);
		File f = new File(filename);
		assertTrue(f.isFile() || f.isDirectory());
		MatrixBlock mbr = IOCompressionTestUtils.read(filename);
		IOCompressionTestUtils.verifyEquivalence(mb, mbr);
	}

	protected static void write(MatrixBlock src, String path) {
		try {
			WriterCompressed.writeCompressedMatrixToHDFS(src, path);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to write file");
		}
	}

	@Test
	public void testWriteAndReadSmallBlen() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514)), 100);
	}

	@Test
	public void testWriteAndReadSmallBlenBiggerClen() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 51, 1, 3, 1.0, 2514)), 50);
	}

	@Test
	public void testWriteAndReadSmallBlenBiggerClenOnly() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 51, 1, 3, 1.0, 2514)), 50);
	}

	@Test
	public void testWriteAndReadSmallBlenBiggerClenMultiBlock() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(50, 124, 1, 3, 1.0, 2514)), 50);
	}

	@Test
	public void testWriteAndReadSmallBlenMultiBlock() throws Exception {
		writeAndRead(TestUtils.ceil(TestUtils.generateTestMatrixBlock(142, 124, 1, 3, 1.0, 2514)), 50);
	}

	protected static void writeAndRead(MatrixBlock mb, int blen) throws Exception {
		String filename = getName();
		WriterCompressed.writeCompressedMatrixToHDFS(mb, filename, blen);
		File f = new File(filename);
		assertTrue(f.isFile() || f.isDirectory());
		MatrixBlock mbr = IOCompressionTestUtils.read(filename);
		IOCompressionTestUtils.verifyEquivalence(mb, mbr);
	}
}
