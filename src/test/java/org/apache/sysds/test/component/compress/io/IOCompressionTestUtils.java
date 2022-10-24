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
import static org.junit.Assert.fail;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.sysds.runtime.compress.io.ReaderCompressed;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class IOCompressionTestUtils {

	final static Object lock = new Object();

	static final AtomicInteger id = new AtomicInteger(0);

	protected static void deleteDirectory(File file) {
		synchronized(IOCompressionTestUtils.lock) {
			for(File subfile : file.listFiles()) {
				if(subfile.isDirectory())
					deleteDirectory(subfile);
				subfile.delete();
			}
			file.delete();
		}
	}

	public static String getName(String nameBeginning) {
		return nameBeginning + "testWrite" + id.incrementAndGet() + ".cla";
	}

	protected static void verifyEquivalence(MatrixBlock a, MatrixBlock b) {
		assertEquals("Nrow is not equivalent", a.getNumRows(), b.getNumRows());
		assertEquals("NCol is not equivalent", a.getNumColumns(), b.getNumColumns());
		assertEquals("Sum is not equivalent", a.sum(), b.sum(), 0.0001);
		assertEquals("Min is not equivalent", a.min(), b.min(), 0.0001);
		assertEquals("Max is not equivalent", a.max(), b.max(), 0.0001);
		// assertTrue("Memory size is not equivalent", a.getInMemorySize() > b.getInMemorySize());
		// assertTrue("Disk size is not equivalent", a.getExactSizeOnDisk() > b.getExactSizeOnDisk());
	}

	public static MatrixBlock read(String path) {
		try {
			return ReaderCompressed.readCompressedMatrixFromHDFS(path);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to read file");
			return null;
		}
	}
}
