package org.apache.sysds.test.functions.federated.io;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.federated.FederatedTestObjectConstructor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCompressionTest extends AutomatedTestBase {

    private static final Log LOG = LogFactory.getLog(FederatedCompressionTest.class.getName());
    private final static String TEST_DIR = "functions/federated/io/";
    private final static String TEST_NAME = "FederatedCompressionTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedCompressionTest.class.getSimpleName() + "/";
    private final static int blocksize = 1024;
    private final static String OUTPUT_NAME = "Z";
    private final static String TEST_CONF_FOLDER = SCRIPT_DIR + TEST_DIR + "config/";

    @Parameterized.Parameter()
    public String compressionStrategy = "Zlib";
    @Parameterized.Parameter(1)
    public int dim = 3;
    @Parameterized.Parameter(2)
    public long[][] begins;
    @Parameterized.Parameter(3)
    public long[][] ends;

    protected String getTestDir() {
        return "functions/federated/io/";
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                // {compressionStrategy, dim, begins, ends}
                {"Zlib", 3, new long[][] {new long[] {0, 0}}, new long[][] {new long[] {3, 3}}},
        });
    }

    @Test
    public void testFederatedReadWriteCompressionStrategies() {
        federatedReadWriteCompression();
    }

    public void federatedReadWriteCompression() {
        System.out.println("CompressionStrategy: " + compressionStrategy);
        System.out.println("Dim: " + dim);
        Types.ExecMode oldPlatform = setExecMode(Types.ExecType.CP);
        getAndLoadTestConfiguration(TEST_NAME);
        //setOutputBuffering(true);

        // empty script name because we don't execute any script, just start the worker

        fullDMLScriptName = "";
        int port1 = getRandomAvailablePort();
        Thread t1 = startLocalFedWorkerThread(port1);
        String host = "localhost";

        try {
            double[][] X1 = new double[][] {new double[] {1, 2, 3}, new double[] {4, 5, 6}, new double[] {7, 8, 9}};
            MatrixCharacteristics mc = new MatrixCharacteristics(dim, dim, blocksize, dim * dim);
            writeCSVMatrix("X1", X1, false, mc);

            // Thread.sleep(10000);
            MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(dim, dim, blocksize, host, begins,
                    ends, new int[] {port1}, new String[] {input("X1")}, input("X.json"));
            writeInputFederatedWithMTD("X.json", fed, null);

            // Run reference dml script with normal matrix
            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/" + TEST_NAME + "1Reference.dml";
            programArgs = new String[] {"-stats", "-nvargs", "fedmatrix=" + input("X1"), "out=" + expected(OUTPUT_NAME)};

            runTest(null);

            // LOG.debug(refOut);

            // Run federated
            fullDMLScriptName = SCRIPT_DIR + "functions/federated/io/" + TEST_NAME + ".dml";
            programArgs = new String[] {"-stats", "-nvargs", "fedmatrix=" + input("X.json"), "out=" + output(OUTPUT_NAME)};
            runTest(null);

            HashMap<MatrixValue.CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
            HashMap<MatrixValue.CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
            TestUtils.compareMatrices(fedResults, refResults, 0, "Fed", "Ref");

            /*Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
            // Verify output
            Assert.assertEquals(Double.parseDouble(refOut.split("\n")[0]), Double.parseDouble(out.split("\n")[0]),
                    0.00001);*/
        } catch (Exception e) {
            e.printStackTrace();
            Assert.assertTrue(false);
        } finally {
            resetExecMode(oldPlatform);
        }

        TestUtils.shutdownThreads(t1);
    }

    /**
     * Override default configuration with custom test configuration to ensure
     * scratch space and local temporary directory locations are also updated.
     */
    @Override
    protected File getConfigTemplateFile() {
        // Instrumentation in this test's output log to show custom configuration file used for template.
        LOG.info("This test case overrides default configuration with " + new File(TEST_CONF_FOLDER + compressionStrategy + "CompressionConfig.xml").getPath());
        return new File(TEST_CONF_FOLDER + compressionStrategy + "CompressionConfig.xml");
    }
}
