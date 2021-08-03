package org.apache.sysds.test.functions.pipelines;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class BuiltinExecutePipelineTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "executePipelineTest";
	private final static String TEST_CLASS_DIR = SCRIPT_DIR + BuiltinExecutePipelineTest.class.getSimpleName() + "/";

	private static final String RESOURCE = SCRIPT_DIR+"functions/pipelines/";
	private static final String DATA_DIR = DATASET_DIR+ "pipelines/";

	private final static String DIRTY = DATA_DIR+ "dirty.csv";
	private final static String META = RESOURCE+ "meta/meta_census.csv";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R"}));
	}

	@Test
	public void testEvalPipClass() {
		execPip(Types.ExecMode.SINGLE_NODE);
	}

	private void execPip(Types.ExecMode et) {

		//		setOutputBuffering(true);
		String HOME = SCRIPT_DIR+"functions/pipelines/" ;
		Types.ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME1));
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-stats", "-exec", "singlenode", "-args", DIRTY, META, output("O")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			//expected loss smaller than default invocation
			Assert.assertTrue(TestUtils.readDMLBoolean(output("O")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}


	public static void main(String[] args) {
		String s = null;
		System.out.println("length is "+s.length());
	}
}
