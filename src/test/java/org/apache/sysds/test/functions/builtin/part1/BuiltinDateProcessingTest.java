package org.apache.sysds.test.functions.builtin.part1;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
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

    @Test
	public void DateProcessingTest0() { //simple conversion to dominant format
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 0, timeformatToAdd = s
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 0, "s", ref);
	}
	@Test
	public void DateProcessingTest1() { //simple conversion to dominant format + 1 day added
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2003", "05-08-2004", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"04-06-2003", "05-07-2003", "06-08-2004", "07-06-2006"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 1, timeformatToAdd = d
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 1, "d", ref);
	}

	@Test
	public void DateProcessingTest2() { //simple conversion to number (timestamp in milliseconds)
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003", "04-07-2004 04:04:04", "05-08-2004", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"1054591200000", "1088906644000", "1091656800000", "1149567360000"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable0_ref_test1);
		//convertToNumber = True, convertToDominant = False, valToAdd = 0
		runDateProcessingTest(f, ExecType.CP, "TRUE", "FALSE", 0, "s", ref);
	}

	@Test
	public void DateProcessingTest3() { //no conversion only time added (3 hours)
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"03-06-2003 23:04:04", "04-07-2003 04:04:04", "05-08-2004 10:04:04", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4"},
            {"1", "2", "3", "4"},
            {"04-06-2003 02:04:04", "04-07-2003 07:04:04", "05-08-2004 13:04:04", "06-06-2006 09:16"}
        };

		FrameBlock f = generateRandomFrameBlock(4, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(4, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = False, valToAdd = 3, timeformatToAdd = H
		runDateProcessingTest(f, ExecType.CP, "FALSE", "FALSE", 3, "H", ref);
	}

	

	@Test
	public void DateProcessingTest4() { //conversion to dominant
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"03-06-2003 00:00", "01-01-2022 01:01", "05-08-2004 00:00", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"}
        };

		FrameBlock f = generateRandomFrameBlock(6, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(6, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 0
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 0, "s", ref);
	}

	@Test
	public void DateProcessingTest5() { //conversion to dominant + time added on minimal frame
		
		
		String[][] testtable0 = new String[][]{
            {"pseudo_value1",},
            {"1"},
            {"03-06-2003 00:00"}
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1"},
            {"1"},
            {"03-06-2003 03:00"}
        };

		FrameBlock f = generateRandomFrameBlock(1, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(1, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 10800 (3 hours in seconds), timeformatToAdd = s
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 10800, "s", ref);
	}
	
	@Test
	public void DateProcessingTest6() { //conversion to dominant + time added
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"03-06-2003 03:00", "01-01-2022 04:01", "05-08-2004 03:00", "15-06-2012 14:12","06-07-2006 20:16", "06-06-2006 09:16"}
        };

		FrameBlock f = generateRandomFrameBlock(6, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(6, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 10800000 (3 hours in milliseconds), timeformatToAdd = ms
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 10800000, "ms", ref);
	}

	@Test
	public void DateProcessingTest7() { //conversion to dominant - time substracted
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"}
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"1", "2", "3", "4", "5",  "6"},
            {"02-06-2003 20:00", "31-12-2021 21:01", "04-08-2004 20:00", "15-06-2012 07:12","06-07-2006 13:16", "06-06-2006 02:16"}
        };

		FrameBlock f = generateRandomFrameBlock(6, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(6, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = -4, timeformatToAdd = H
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", -4, "H", ref);
	}

	@Test
	public void DateProcessingTest8() { //conversion to dominant - time substracted on frame with dates in multiple columns (making sure it finds the date column)
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
			{"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"},
            {"another_pseudo_value1", "another_pseudo_value2", "another_pseudo_value3", "another_pseudo_value4",  "another_pseudo_value5", "another_pseudo_value6"},
			{"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12", "5",  "6"}
            
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"02-06-2003 20:00", "31-12-2021 21:01", "04-08-2004 20:00", "15-06-2012 07:12","06-07-2006 13:16", "06-06-2006 02:16"},
			{"another_pseudo_value1", "another_pseudo_value2", "another_pseudo_value3", "another_pseudo_value4",  "another_pseudo_value5", "another_pseudo_value6"},
			{"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12", "5",  "6"}
        };

		FrameBlock f = generateRandomFrameBlock(6, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(6, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = -14400000 (4 hours in milliseconds), timeformatToAdd = ms
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", -14400000, "ms", ref);
	}

	@Test
	public void DateProcessingTest9() {
		String[][] testtable0 = new String[][]{
			{"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12", "pseudo_value1", "pseudo_value2"},
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
			{"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12","06-07-2006 17:16", "06-06-2006 06:16"},
            {"1", "2", "3", "4", "5",  "6"}
			
            
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
            {"03-06-2003", "01-01-2022 01:01", "05-08-2004", "15-06-2012 11:12", "pseudo_value1", "pseudo_value2"},
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "05-08-2004"},
            {"07-07-2004 00:00", "05-02-2023 01:01", "09-09-2005 00:00", "20-07-2013 11:12","10-08-2007 17:16", "11-07-2007 06:16"},
			{"1", "2", "3", "4", "5",  "6"}
			
        };

		FrameBlock f = generateRandomFrameBlock(6, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(6, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = 400, timeformatToAdd = d
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", 400, "d", ref);
	}

	@Test
	public void DateProcessingTest10() { //conversion to dominant - time substracted on date column with multiple different date formats
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
			//dd-MM-yyyy, yyyy-MM-dd HH:mm, yyyyMMddHHmmss, dd-MM-yyyy HH:mm, dd-MM-yyyy HH:mm, MM/dd/yyyy HH:mm:ss, dd MMM yyyy HH:mm
			{"03-06-2003", "2022-01-01 01:01", "20040805000000", "15-06-2012 11:12","06-07-2006 17:16", "06/07/2006 06:16:44", "04 Mar 2010 12:20"},
            {"1", "2", "3", "4", "5",  "6", "05-08-2004"}
			
            
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
            {"02-06-2003 20:00", "31-12-2021 21:01", "04-08-2004 20:00", "15-06-2012 07:12","06-07-2006 13:16", "07-06-2006 02:16", "04-03-2010 08:20"},
			{"1", "2", "3", "4", "5",  "6", "05-08-2004"}
			
        };

		FrameBlock f = generateRandomFrameBlock(7, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(7, 3, testtable0_ref_test1);
		//convertToNumber = False, convertToDominant = True, valToAdd = -14400 (4 hours in seconds), timeformatToAdd = s
		runDateProcessingTest(f, ExecType.CP, "FALSE", "TRUE", -14400, "s", ref);
	}

	@Test
	public void DateProcessingTest11() {//conversion to number on date column with multiple different date formats
		String[][] testtable0 = new String[][]{
            {"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
			//dd-MM-yyyy, yyyy-MM-dd HH:mm, yyyyMMddHHmmss, dd-MM-yyyy HH:mm, dd-MM-yyyy HH:mm, MM/dd/yyyy HH:mm:ss, dd MMM yyyy HH:mm
			{"03-06-2003", "2022-01-01 01:01", "20040805000000", "15-06-2012 11:12","06-07-2006 17:16", "06/07/2006 06:16:44", "04 Mar 2010 12:20"},
            {"1", "2", "3", "4", "5",  "6", "05-08-2004"}
			
            
        };

        String[][] testtable0_ref_test1 = new String[][]{ 
			{"pseudo_value1", "pseudo_value2", "pseudo_value3", "pseudo_value4",  "pseudo_value5", "08-09-2013 22:12", "05-08-2004"},
            {"1054591200000", "1640995260000", "1091656800000", "1339751520000","1152198960000", "1149653804000", "1267701600000"},
			{"1", "2", "3", "4", "5",  "6", "05-08-2004"}
			
        };

		FrameBlock f = generateRandomFrameBlock(7, 3, testtable0);
        FrameBlock ref = generateRandomFrameBlock(7, 3, testtable0_ref_test1);
		//convertToNumber = True, convertToDominant = False, valToAdd = 0
		runDateProcessingTest(f, ExecType.CP, "TRUE", "FALSE", 0, "s", ref);
	}

	private void runDateProcessingTest(FrameBlock test_frame, ExecType et, String convertToNumber, String convertToDominant, int valToAdd, 
		String timeformatToAdd, FrameBlock reference)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"),
				"Y=" + output("Y"), "convertToNumber=" + convertToNumber, "convertToDominant=" + convertToDominant, "valToAdd=" + valToAdd, "timeformatToAdd=" + timeformatToAdd};

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
				//System.out.println("ref: " + reference.get(row, col) + ", out: " + results.get(row, col));
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
