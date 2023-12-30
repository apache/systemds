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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertEquals;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.io.WriterBinaryBlockParallel;
import org.junit.Test;

public class NumPartsFilesTest {

	private final Path path;
	private final FileSystem fs;

	public NumPartsFilesTest() throws Exception {
		path = new Path("/tmp/test.someEnding");
		fs = path.getFileSystem(ConfigurationManager.getCachedJobConf());
	}

	@Test
	public void numPartsTest1() {

		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 1000, 1000, 1000, 1000 * 1000);
		assertEquals(1, p);
	}

	@Test
	public void numPartsTest2() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 1200, 1000, 1000, 1000 * 1000);
		assertEquals(2, p);
	}

	@Test
	public void numPartsTest3() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 10000, 1000, 1000, 10000L * 1000);
		assertEquals(10, p);
	}

	@Test
	public void numPartsTest4() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 100000, 1000, 1000, 100000L * 1000);
		assertEquals(100, p);
	}

	@Test
	public void numPartsTest5() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 1000000, 1000, 1000, 1000000L * 1000);
		assertEquals(1000, p);
	}

	@Test
	public void numPartsTest6() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 10000000L, 1000, 1000, 10000000L * 1000);
		assertEquals(10000, p);
	}

	@Test
	public void numPartsTest7() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 100000000L, 1000, 1000, 100000000L * 1000);
		assertEquals(100000, p);
	}

	@Test
	public void numPartsTest8() {
		int p = WriterBinaryBlockParallel.numPartsFiles(fs, 1000000000L, 1000, 1000, 1000000000L * 1000);
		assertEquals(1000000, p);
	}
}
