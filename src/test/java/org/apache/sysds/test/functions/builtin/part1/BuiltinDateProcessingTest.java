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

import static org.junit.Assert.assertEquals;

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

public class BuiltinDateProcessingTest extends AutomatedTestBase {
    private final static String TEST_NAME = "date_processing";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDateProcessingTest.class.getSimpleName() + "/";

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

	static enum TestType {
		DOMINANT,
		DOMINANT_DAY,
		TIMESTAMP,
		ADD_HOURS,
		SUB_MON
	}

    @Test //simple conversion to dominant format
	public void DateProcessingTest0() {

        String[][] testtable_ref = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006"}
        };
		String[][] testtable = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
			{"1", "2", "3", "4"},
			{"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006 06:16"}
		};
		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = False, convertToDominant = True, valToAdd = 0
		runDateProcessingTest(TestType.DOMINANT, f, ExecType.CP, 0, ref);
	}

	@Test //simple conversion to dominant format + 1 day added
	public void DateProcessingTest1() {
		String[][] testtable = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006 06:16"}
        };
        String[][] testtable_ref = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"04-06-2003", "05-07-2003", "06-08-2004", "07-06-2006"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = False, convertToDominant = True, valToAdd = 1, timeformatToAdd = d
		runDateProcessingTest(TestType.DOMINANT_DAY, f, ExecType.CP, 1,  ref);
	}
//
	@Test //simple conversion to number (timestamp in milliseconds)
	public void DateProcessingTest2() {
		String[][] testtable = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2004 04:04:04", "05-08-2004", "06-06-2006 06:16"}
        };
        String[][] testtable_ref = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"1054598400000", "1088913844000", "1091664000000", "1149574560000"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = True, convertToDominant = False, valToAdd = 0
		runDateProcessingTest(TestType.TIMESTAMP, f, ExecType.CP,  0,  ref);
	}

	@Test //simple conversion to number (timestamp in milliseconds)
	public void DateProcessingTest2SP() {
		String[][] testtable = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
			{"1", "2", "3", "4"},
			{"03-06-2003", "04-07-2004 04:04:04", "05-08-2004", "06-06-2006 06:16"}
		};
		String[][] testtable_ref = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
			{"1", "2", "3", "4"},
			{"1054598400000", "1088913844000", "1091664000000", "1149574560000"}
		};

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
		FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = True, convertToDominant = False, valToAdd = 0
		runDateProcessingTest(TestType.TIMESTAMP, f, ExecType.SPARK,  0,  ref);
	}

	@Test //no conversion only time added (3 hours)
	public void DateProcessingTest3() {
		String[][] testtable = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003 23:04:04", "04-07-2003 04:04:04", "05-08-2004 10:04:04", "06-06-2006 06:16"}
        };
        String[][] testtable_ref = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"04-06-2003 02:04:04", "04-07-2003 07:04:04", "05-08-2004 13:04:04", "06-06-2006 09:16"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = False, convertToDominant = False, valToAdd = 3, timeformatToAdd = H
		runDateProcessingTest(TestType.ADD_HOURS, f, ExecType.CP, 3,  ref);
	}

	@Test //no conversion only time added (3 hours)
	public void DateProcessingTest3SPARK() {
		String[][] testtable = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
			{"1", "2", "3", "4"},
			{"03-06-2003 23:04:04", "04-07-2003 04:04:04", "05-08-2004 10:04:04", "06-06-2006 06:16"}
		};
		String[][] testtable_ref = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
			{"1", "2", "3", "4"},
			{"04-06-2003 02:04:04", "04-07-2003 07:04:04", "05-08-2004 13:04:04", "06-06-2006 09:16"}
		};

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable);
		FrameBlock ref = generateRandomFrameBlock(4, 3, testtable_ref);
		//convertToNumber = False, convertToDominant = False, valToAdd = 3, timeformatToAdd = H
		runDateProcessingTest(TestType.ADD_HOURS, f, ExecType.SPARK, 3,  ref);
	}

	@Test //no conversion, only substraction of 3 months
	public void DateProcessingTest12() {
		String[][] testtable = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
			{"03-06-2003", "04 Sep 2006 09:06", "20040805121518", "15-06-2012 11:12","10 Oct 2010 10:10",
				"06/07/2006 06:16:44", "06 Mar 2010 12:20"},
            {"1", "2", "3", "4", "5",  "6", "05-08-2004"}
        };
        String[][] testtable_ref = new String[][]{
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
            {"03-03-2003", "04 Jun 2006 09:06", "20040505121518", "15-03-2012 11:12","10 Jul 2010 10:10",
				"03/07/2006 06:16:44", "06 Dec 2009 12:20"},
			{"1", "2", "3", "4", "5",  "6", "05-08-2004"}
        };

		FrameBlock f = generateRandomFrameBlock(7, 3, testtable);
        FrameBlock ref = generateRandomFrameBlock(7, 3, testtable_ref);
		//convertToNumber = False, convertToDominant = False, valToAdd = -3 , timeformatToAdd = M
		runDateProcessingTest(TestType.SUB_MON,f, ExecType.CP,  -3,  ref);
	}

	private void runDateProcessingTest(TestType testName, FrameBlock test_frame, ExecType et, int valToAdd, FrameBlock reference)
	{
		Types.ExecMode platformOld = setExecMode(et);
		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"),
				"Y=" + output("Y"),	"valToAdd=" + valToAdd, "test="+testName.toString()};
			FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV).
					writeFrameToHDFS(test_frame, input("F"), test_frame.getNumRows(), test_frame.getNumColumns());
			System.out.println("running test ...");
			runTest(true, false, null, -1);
			System.out.println("Done. Validating results.");
			FrameBlock outputFrame = readDMLFrameFromHDFS("Y", Types.FileFormat.CSV);

            validateResults(reference, outputFrame);
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private void validateResults (FrameBlock reference, FrameBlock results){
		int cols = results.getNumColumns();
		int rows = results.getNumRows();

		for (int col = 0; col < cols; col++) {
			for (int row = 0; row < rows; row++) {
				assertEquals(reference.get(row, col), results.get(row, col));
				System.out.println("ref: " + reference.get(row, col) + ", out: " + results.get(row, col));
			}
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
}
