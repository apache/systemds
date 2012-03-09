package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Amol Ghoting
 */
public class SequenceMiner extends AutomatedTestBase {

	private final static String TEST_SEQMINER = "SeqMiner"; 
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("SeqMiner", new TestConfiguration(baseDirectory, "SeqMiner", new String[] { "fseq", "sup" }));
	}

	@Test
	public void testSequenceMiner() {
		
		int rows = 5;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SeqMiner");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		/* This is for running the junit test by constructing the arguments directly */
		String SEQMINER_HOME = baseDirectory;
		dmlArgs = new String[]{"-f", SEQMINER_HOME + TEST_SEQMINER + ".dml",
				               "-args", "\"" + SEQMINER_HOME + INPUT_DIR + "M" + "\"", 
				                        Integer.toString(rows), Integer.toString(cols), 
				                        "\"" + SEQMINER_HOME + OUTPUT_DIR + "fseq" + "\"",
				                        "\"" + SEQMINER_HOME + OUTPUT_DIR + "sup" + "\""};
		dmlArgsDebug = new String[]{"-f", SEQMINER_HOME + TEST_SEQMINER + ".dml", "-d", 
				                    "-args", "\"" + SEQMINER_HOME + INPUT_DIR + "M" + "\"", 
                                             Integer.toString(rows), Integer.toString(cols), 
                                             "\"" + SEQMINER_HOME + OUTPUT_DIR + "fseq" + "\"",
                                             "\"" + SEQMINER_HOME + OUTPUT_DIR + "sup" + "\""};
		
		double[][] M = {{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}};
		
		
		writeInputMatrix("M", M);
		
		loadTestConfiguration(config);

		// no expected number of M/R jobs are calculated, set to default for now
		runTest(true, false, null, -1);

		
		checkForResultExistence();
	}
}
