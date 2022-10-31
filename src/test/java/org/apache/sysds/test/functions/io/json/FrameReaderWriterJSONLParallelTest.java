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
package org.apache.sysds.test.functions.io.json;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.wink.json4j.JSONException;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderJSONLParallel;
import org.apache.sysds.runtime.io.FrameWriterJSONLParallel;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

public class FrameReaderWriterJSONLParallelTest 
{
	private static final String FILENAME_SINGLE = "target/testTemp/functions/data/FrameJSONLParallelTest/testFrameBlock.json";

	@Test
	public void testWriteReadFrameBlockSingleSingleFromHDFS() throws IOException, JSONException {
		testWriteReadFrameBlockComplete(1,1, 4669201);
	}

	@Test
	public void testWriteReadFrameBlockSingleMultipleFromHDFS() throws IOException, JSONException {
		testWriteReadFrameBlockComplete(1,23, 4669201);
	}

	@Test
	public void testWriteReadFrameBlockMultipleSingleFromHDFS() throws IOException, JSONException {
		testWriteReadFrameBlockComplete(21,1, 4669201);
	}

	@Test
	public void testWriteReadFrameBlockMultipleMultipleFromHDFS() throws IOException, JSONException {
		testWriteReadFrameBlockComplete(42,35, 4669201);
	}

	@Test
	public void testWriteReadFrameBlockMultipleMultipleSmallFromHDFS() throws IOException, JSONException { 
		testWriteReadFrameBlockComplete(694,164, 4669201);
	}

	@Test
	public void testWriteReadFrameBlockMultipleMultipleMediumFromHDFS() throws IOException, JSONException { 
		testWriteReadFrameBlockComplete(30424,327, 4669201);
	}

	public void testWriteReadFrameBlockComplete(int rows, int cols, long seed) throws IOException, JSONException {
		Pair<FrameBlock, FrameBlock> writeReadPair = testWriteReadFullFrameBlockFromHDFS(rows, cols, seed);
		String[][] strWriteBlock = DataConverter.convertToStringFrame(writeReadPair.getLeft());
		String[][] strReadBlock = DataConverter.convertToStringFrame(writeReadPair.getRight());
		TestUtils.compareFrames(strWriteBlock, strReadBlock, rows, cols);
	}

	public Pair<FrameBlock,FrameBlock> testWriteReadFullFrameBlockFromHDFS(int rows, int cols,
		Types.ValueType[] schema, Map<String, Integer> schemaMap, Random random)
			throws IOException, JSONException 
	{
		FrameWriterJSONLParallel frameWriterJSONL = new FrameWriterJSONLParallel();
		FrameReaderJSONLParallel frameReaderJSONL = new FrameReaderJSONLParallel();

		// Generate Random frameBlock to be written
		FrameBlock writeBlock = TestUtils.generateRandomFrameBlock(rows, cols, schema, random);

		// Write FrameBlock
		frameWriterJSONL.writeFrameToHDFS(writeBlock, FILENAME_SINGLE, schemaMap, rows,cols);

		// Read FrameBlock
		FrameBlock readBlock = frameReaderJSONL.readFrameFromHDFS(FILENAME_SINGLE, schema, schemaMap, rows, cols);

		return Pair.of(writeBlock, readBlock);

	}

	public Pair<FrameBlock,FrameBlock> testWriteReadFullFrameBlockFromHDFS(int rows, int cols, long seed)
		throws IOException, JSONException
	{
		Random random = new Random(seed);
		Map<String, Integer> schemaMap = TestUtils.generateRandomSchemaMap(cols, random);
		Types.ValueType[] schema = TestUtils.generateRandomSchema(cols, random);
		return testWriteReadFullFrameBlockFromHDFS(rows, cols, schema, schemaMap, random);
	}
}
