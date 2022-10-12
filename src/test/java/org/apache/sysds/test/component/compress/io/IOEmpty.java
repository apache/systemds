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
import org.apache.sysds.runtime.compress.io.WriterCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.AfterClass;
import org.junit.Test;

public class IOEmpty {

	protected static final Log LOG = LogFactory.getLog(IOTest.class.getName());
	
	final static String nameBeginning = "src/test/java/org/apache/sysds/test/component/compress/io/files"
		+ IOEmpty.class.getSimpleName() + "/";

	public IOEmpty() {
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
	public void writeEmpty() {
		String n = getName();
		write(n, 10, 10, 1000);
		File f = new File(n);
		assertTrue(f.isFile());
	}

	@Test
	public void writeEmptyAndRead() {
		String n = getName();
		write(n, 10, 10, 1000);
		MatrixBlock mb = IOCompressionTestUtils.read(n);
		IOCompressionTestUtils.verifyEquivalence(mb, new MatrixBlock(10, 10, 0.0));
	}

	@Test
	public void writeEmptyMultiBlock() {
		String n = getName();
		write(n, 1000, 10, 100);
		File f = new File(n);
		assertTrue(f.isDirectory() || f.isFile());
	}

	@Test
	public void writeEmptyAndReadMultiBlock() {
		String n = getName();
		write(n, 1000, 10, 100);
		File f = new File(n);
		assertTrue(f.isDirectory() || f.isFile());
		MatrixBlock mb = IOCompressionTestUtils.read(n);
		IOCompressionTestUtils.verifyEquivalence(mb, new MatrixBlock(1000, 10, 0.0));
	}

	protected static void write(String path, int nRows, int nCols, int blen) {
		try {
			WriterCompressed w = WriterCompressed.create(null);
			w.writeEmptyMatrixToHDFS(path, nRows, nCols, blen);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to write file");
		}
	}
}
