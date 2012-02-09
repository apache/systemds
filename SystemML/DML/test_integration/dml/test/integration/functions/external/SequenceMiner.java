package dml.test.integration.functions.external;

import org.junit.Test;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

/**
 * 
 * @author Amol Ghoting
 */
public class SequenceMiner extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put("SeqTest", new TestConfiguration("SeqTest", new String[] { "fseq", "sup" }));
	}

	@Test
	public void testSequenceMiner() {
		
		int rows = 5;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SeqTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		double[][] M = {{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}};
		
		
		writeInputMatrix("M", M);
		
		loadTestConfiguration(config);

		runTest();

		
		checkForResultExistence();
	}
}
