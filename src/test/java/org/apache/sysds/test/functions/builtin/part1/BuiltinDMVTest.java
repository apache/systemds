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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

public class BuiltinDMVTest extends AutomatedTestBase {

	private final static String TEST_NAME = "disguisedMissingValue";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDMVTest.class.getSimpleName() + "/";

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void NormalStringFrameTest() {
		FrameBlock f = generateRandomFrameBlock(1000, 4,null);
		String[] disguised_values = new String[]{"?", "9999", "?", "9999"};
		ArrayList<List<Integer>> positions = getDisguisedPositions(f, 4, disguised_values);
		runMissingValueTest(f, ExecType.CP, 0.8, "DMV", positions);
	}

	@Test
	public void PreDefinedStringsFrameTest() {
		String[] testarray0 = new String[]{"77","77","55","89","43", "99", "46"}; // detect Weg
		String[] testarray1 = new String[]{"8010","9999","8456","4565","89655", "86542", "45624"}; // detect ?
		String[] testarray2 = new String[]{"David K","Valentin E","Patrick L","VEVE","DK", "VE", "PL"}; // detect 45
		String[] testarray3 = new String[]{"3.42","45","0.456",".45","4589.245", "97", "33"}; // detect ka
		String[] testarray4 = new String[]{"99","123","158","146","158", "174", "201"}; // detect 9999

		String[][] teststrings = new String[][]{testarray0, testarray1, testarray2, testarray3, testarray4};
		FrameBlock f = generateRandomFrameBlock(7, 5, teststrings);
		String[] disguised_values = new String[]{"Patrick-Lovric-Weg-666", "?", "45", "ka", "9999"};
		ArrayList<List<Integer>> positions = getDisguisedPositions(f, 1, disguised_values);
		runMissingValueTest(f, ExecType.CP, 0.7,"NA", positions);
	}

	@Test
	public void PreDefinedDoubleFrame() {
		Double[] test_val = new Double[10000];
		for(int i = 0; i < test_val.length; i++) {
			test_val[i] = TestUtils.getPositiveRandomDouble();
		}
		String[] test_string = new String[test_val.length];
		for(int j = 0; j < test_val.length; j++) {
			test_string[j] = test_val[j].toString();
		}

		String[][] teststrings = new String[][]{test_string};
		FrameBlock f = generateRandomFrameBlock(test_string.length, 1, teststrings);
		String[] disguised_values = new String[]{"9999999999"};
		ArrayList<List<Integer>> positions = getDisguisedPositions(f, 10, disguised_values);
		runMissingValueTest(f, ExecType.CP, 0.6, "-1", positions);
	}

	private void runMissingValueTest(FrameBlock test_frame, ExecType et, Double threshold, String replacement,
		ArrayList<List<Integer>> positions)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"),
				"O=" + output("O"), "threshold=" + threshold, "replacement=" + replacement};

			FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV).
					writeFrameToHDFS(test_frame, input("F"), test_frame.getNumRows(), test_frame.getNumColumns());

			runTest(true, false, null, -1);

			FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);

			for(int i = 0; i < positions.size(); i++) {
				String[] output = (String[]) outputFrame.getColumnData(i);
				for(int j = 0; j < positions.get(i).size(); j++) {
					if(replacement.equals("NaN")) 
						TestUtils.compareScalars(null, output[positions.get(i).get(j)]);
					else 
						TestUtils.compareScalars(replacement, output[positions.get(i).get(j)]);
				}
			}
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static FrameBlock generateRandomFrameBlock(int rows, int cols, String[][] defined_strings)
	{
		Types.ValueType[] schema = new Types.ValueType[cols];
		for(int i = 0; i < cols; i++) {
			schema[i] = Types.ValueType.STRING;
		}

		if(defined_strings != null) {
			String[] names = new String[cols];
			for(int i = 0; i < cols; i++)
				names[i] = schema[i].toString();
			FrameBlock frameBlock = new FrameBlock(schema, names);
			frameBlock.ensureAllocatedColumns(rows);
			for(int row = 0; row < rows; row++)
				for(int col = 0; col < cols; col++)
					frameBlock.set(row, col, defined_strings[col][row]);
			return frameBlock;
		}
		return TestUtils.generateRandomFrameBlock(rows, cols, schema ,TestUtils.getPositiveRandomInt());
	}

	private static ArrayList<List<Integer>> getDisguisedPositions(FrameBlock frame,
		int amountValues, String[] disguisedValue)
	{
		ArrayList<List<Integer>> positions = new ArrayList<>();
		int counter;
		for(int i = 0; i < frame.getNumColumns(); i++) {
			counter = 0;
			List<Integer> arrayToFill = new ArrayList<>();
			while(counter < frame.getNumRows() && counter < amountValues) {
				int position = TestUtils.getPositiveRandomInt() % frame.getNumRows();
				while(counter != 0 && arrayToFill.contains(position)) {
					position = (position + TestUtils.getPositiveRandomInt() + 5) % frame.getNumRows();
				}
				arrayToFill.add(position);
				if(disguisedValue.length > 1)
					frame.set(position, i, disguisedValue[i]);
				else if (disguisedValue.length == 1)
					frame.set(position, i, disguisedValue[0]);
				counter++;
			}
			positions.add(i, arrayToFill);
		}

		return positions;
	}
}
