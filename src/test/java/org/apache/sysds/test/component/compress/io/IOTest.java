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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.io.ReaderCompressed;
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Test;

public class IOTest {

	protected static final Log LOG = LogFactory.getLog(IOTest.class.getName());

	final static Object lock = new Object();

	final static String nameBeginning = "src/test/java/org/apache/sysds/test/component/compress/io/files/";

	static AtomicInteger id = new AtomicInteger(0);

	public IOTest() {
		synchronized(lock) {
			new File(nameBeginning).mkdirs();
		}
	}

	private static void deleteDirectory(File file) {
		for(File subfile : file.listFiles()) {
			if(subfile.isDirectory())
				deleteDirectory(subfile);
			subfile.delete();
		}
		file.delete();
	}

	@AfterClass
	public static void cleanup() {
		deleteDirectory(new File(nameBeginning));
	}

	public static String getName() {
		return nameBeginning + "testWrite" + id.incrementAndGet() + ".cla";
	}

	@Test
	public void testWrite() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514));
		write(mb, getName());
	}

	@Test
	public void testWriteAlreadyCompressed() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514));
		MatrixBlock mb2 = CompressedMatrixBlockFactory.compress(mb).getLeft();
		write(mb2, getName());
	}

	@Test
	public void testWriteAndRead() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1000, 3, 1, 3, 1.0, 2514));

		String filename = getName();
		write(mb, filename);
		MatrixBlock mbr = read(filename);

		assertEquals(mb.sum(), mbr.sum(), 0.0001);
		assertEquals(mb.min(), mbr.min(), 0.0001);
		assertEquals(mb.max(), mbr.max(), 0.0001);
		assertEquals(mb.getNumRows(), mbr.getNumRows());
		assertEquals(mb.getNumColumns(), mbr.getNumColumns());
		assertTrue(mb.getInMemorySize() > mbr.getInMemorySize());
		assertTrue(mb.getExactSizeOnDisk() > mbr.getExactSizeOnDisk());

	}

	@Test(expected = DMLCompressionException.class)
	public void testWriteNotCompressable() throws Exception {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(3, 3, 1, 3, 1.0, 2514));
		WriterCompressed.writeCompressedMatrixToHDFS(mb, getName());
	}

	private static MatrixBlock read(String path) {
		try {
			return ReaderCompressed.readCompressedMatrixFromHDFS(path);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to read file");
			return null;
		}
	}

	private static void write(MatrixBlock src, String path) {
		try {
			WriterCompressed.writeCompressedMatrixToHDFS(src, path);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to write file");
		}
	}
}
