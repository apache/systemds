package org.apache.sysds.test.functions.io.cog;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;



import java.util.HashMap;

public abstract class ReadCOGTest extends COGTestBase {
    protected abstract int getId();

    @Test
    public void testCOG_Seq_CP() {
        runReadCOGTest(getId(), getResult(), Types.ExecMode.SINGLE_NODE, false);
    }

    protected String getInputCOGFileName() {
        return "testCOG_" + getId();
    }

    protected abstract double getResult();
    @Test
    public void testCOG_Parallel_CP() {
        runReadCOGTest(getId(), getResult(), Types.ExecMode.SINGLE_NODE, true);
    }

    protected void runReadCOGTest(int testNumber, double result, Types.ExecMode platform, boolean parallel) {
        Types.ExecMode oldPlatform = rtplatform;
        rtplatform = platform;

        boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if(rtplatform == Types.ExecMode.SPARK)
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;

        boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT; // set to false for debugging maybeee

        try {
            CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
            TestConfiguration config = getTestConfiguration(getTestName());
            loadTestConfiguration(config);

            String HOME = SCRIPT_DIR + TEST_DIR;
            String inputMatrixName = DATASET_DIR + "cog/" + getInputCOGFileName() + ".tif";

            String dmlOutput = output("dml.scalar");

            fullDMLScriptName = HOME + getTestName() + "_" + getScriptId() + ".dml";
            programArgs = new String[] {"-args", inputMatrixName, dmlOutput};

            runTest(true, false, null, -1);

            double dmlScalarOutput = TestUtils.readDMLScalar(dmlOutput);
            TestUtils.compareScalars(dmlScalarOutput, result, eps * getResult());
        }
        finally {
            rtplatform = oldPlatform;
            CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
    }
}

