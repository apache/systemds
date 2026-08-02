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

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

/** Shared helpers for the native Delta frame read/write tests and benchmarks. */
public class DeltaFrameTestUtils {

	private DeltaFrameTestUtils() {
		// utility class
	}

	/** Count the parquet data files under a Delta table directory. */
	public static long countParquet(String tablePath) throws Exception {
		try(Stream<Path> s = Files.walk(new File(tablePath).toPath())) {
			return s.filter(p -> p.toString().endsWith(".parquet")).count();
		}
	}
}
