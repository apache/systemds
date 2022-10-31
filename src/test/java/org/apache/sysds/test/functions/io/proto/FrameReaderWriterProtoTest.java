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

package org.apache.sysds.test.functions.io.proto;

import java.io.IOException;
import java.util.Random;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameReaderWriterProtoTest {

	private static final String FILENAME_SINGLE = "target/testTemp/functions/data/FrameReaderWriterProtoTest/testFrameBlock.proto";
	private static final long SEED = 4669201;

	private FrameWriter frameWriterProto = FrameWriterFactory.createFrameWriter(Types.FileFormat.PROTO);
	private FrameReader frameReaderProto = FrameReaderFactory.createFrameReader(Types.FileFormat.PROTO);

	@Test
	public void testWriteReadFrameBlockWithSinleRowAndSingleColumnFromHDFS() throws IOException {
		testWriteReadFrameBlockWith(1, 1);
	}

	@Test
	public void testWriteReadFrameBlockWithSingleRowAndMultipleColumnsFromHDFS() throws IOException {
		testWriteReadFrameBlockWith(1, 23);
	}

	@Test
	public void testWriteReadFrameBlockWithMultipleRowsAndSingleColumnFromHDFS() throws IOException {
		testWriteReadFrameBlockWith(21, 1);
	}

	@Test
	public void testWriteReadFrameBlockWithSmallMultipleRowsAndMultipleColumnsFromHDFS() throws IOException {
		testWriteReadFrameBlockWith(42, 35);
	}

	@Test
	public void testWriteReadFrameBlockWithMediumMultipleRowsAndMultipleColumnsFromHDFS() throws IOException {
		testWriteReadFrameBlockWith(694, 164);
	}

	public void testWriteReadFrameBlockWith(int rows, int cols) throws IOException {
		final Random random = new Random(SEED);
		Types.ValueType[] schema = TestUtils.generateRandomSchema(cols, random);
		FrameBlock expectedFrame = TestUtils.generateRandomFrameBlock(rows, cols, schema, random);

		frameWriterProto.writeFrameToHDFS(expectedFrame, FILENAME_SINGLE, rows, cols);
		FrameBlock actualFrame = frameReaderProto.readFrameFromHDFS(FILENAME_SINGLE, schema, rows, cols);

		String[][] expected = DataConverter.convertToStringFrame(expectedFrame);
		String[][] actual = DataConverter.convertToStringFrame(actualFrame);

		TestUtils.compareFrames(expected, actual, rows, cols);
	}
}
