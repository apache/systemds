package org.apache.sysds.test.component.resource;

import org.apache.hadoop.io.ShortWritable;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.CloudUtils;
import org.apache.sysds.resource.cost.CostEstimationException;
import org.apache.sysds.resource.enumeration.EnumerationUtils.SolutionPoint;
import org.apache.sysds.resource.enumeration.Enumerator;
import org.apache.sysds.resource.enumeration.GridBasedEnumerator;
import org.apache.sysds.resource.enumeration.PruneBasedEnumerator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import static org.apache.sysds.test.component.resource.ResourceTestUtils.DEFAULT_INSTANCE_INFO_TABLE;
import static org.apache.sysds.test.component.resource.ResourceTestUtils.assertEqualsCloudInstances;

public class PruneBaseEnumeratorTests extends AutomatedTestBase {
    static {
        ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
    }
    private static final String TEST_DIR = "component/resource/";
    private static final String HOME = SCRIPT_DIR + TEST_DIR;
    private static final String TEST_CLASS_DIR = TEST_DIR + CostEstimatorTest.class.getSimpleName() + "/";
    private static HashMap<String, CloudInstance> allInstances;

    @BeforeClass
    public static void setUpBeforeClass() {
        try {
            allInstances = CloudUtils.loadInstanceInfoTable(DEFAULT_INSTANCE_INFO_TABLE, 0.25, 0.08);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void setUp() {}

    @Test
    public void L2SSVMDefaultTest() {
        runTest("Algorithm_L2SVM.dml");
    }

    @Test
    public void L2SSVMLargeInputTest() {
        // input of 80GB
        Tuple2<String, String> mVar = new Tuple2<>("$m", "1000000");
        Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
        runTest("Algorithm_L2SVM.dml", mVar, nVar);
    }

    @Test
    public void LinregDefaultTest() {
        runTest("Algorithm_Linreg.dml");
    }

    @Test
    public void LinregLargeInputTest() {
        // input of 80GB
        Tuple2<String, String> mVar = new Tuple2<>("$m", "1000000");
        Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
        runTest("Algorithm_Linreg.dml", mVar, nVar);
    }

    @Test
    public void PCADefaultTest() {
        runTest("Algorithm_PCA.dml");
    }

    @Test
    public void PCALargeInputTest() {
        // input of 80GB
        Tuple2<String, String> mVar = new Tuple2<>("$m", "1000000");
        Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
        runTest("Algorithm_PCA.dml", mVar, nVar);
    }

    @Test
    public void PNMFDefaultTest() {
        runTest("Algorithm_PNMF.dml");
    }

    @Test
    public void PNMFLargeInputTest() {
        // input of 80GB
        Tuple2<String, String> mVar = new Tuple2<>("$m", "1000000");
        Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
        runTest("Algorithm_PNMF.dml", mVar, nVar);
    }

    @Test
    public void ReadAndWriteTest() {
        Tuple2<String, String> arg1 = new Tuple2<>("$fileA", HOME+"data/A.csv");
        Tuple2<String, String> arg2 = new Tuple2<>("$fileA_Csv", HOME+"data/A_copy.csv");
        Tuple2<String, String> arg3 = new Tuple2<>("$fileA_Text", HOME+"data/A_copy_text.text");
        runTest("ReadAndWrite.dml", arg1, arg2, arg3);
    }

    // Helpers ---------------------------------------------------------------------------------------------------------

    @SafeVarargs
    private void runTest(String scriptFilename, Tuple2<String, String>...args) {
        long startTime, endTime;
        Program program = ResourceTestUtils.compileProgramWithNvargs(HOME + scriptFilename, args);
        // grid-based enumerator for expected value generation
        GridBasedEnumerator gridEnumerator = (GridBasedEnumerator) (new Enumerator.Builder())
                .withRuntimeProgram(program)
                .withAvailableInstances(allInstances)
                .withEnumerationStrategy(Enumerator.EnumerationStrategy.GridBased)
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinCosts)
                .withNumberExecutorsRange(0, 5)
                .build();
        System.out.println("Launching the Grid-Based Enumerator for expected solution");
        startTime = System.currentTimeMillis();
        gridEnumerator.preprocessing();
        gridEnumerator.processing();
        SolutionPoint expectedSolution = gridEnumerator.postprocessing();
        endTime = System.currentTimeMillis();
        System.out.println("Grid-Based Enumerator finished for "+(endTime-startTime)/1000+"s");

        // prune-based enumerator for actual value generation
        PruneBasedEnumerator pruningEnumerator = (PruneBasedEnumerator) (new Enumerator.Builder())
                .withRuntimeProgram(program)
                .withAvailableInstances(allInstances)
                .withEnumerationStrategy(Enumerator.EnumerationStrategy.PruneBased)
                .withOptimizationStrategy(Enumerator.OptimizationStrategy.MinCosts)
                .withNumberExecutorsRange(0, 5)
                .build();
        System.out.println("Launching the Prune-Based Enumerator for testing solution");
        startTime = System.currentTimeMillis();
        pruningEnumerator.preprocessing();
        pruningEnumerator.processing();
        SolutionPoint actualSolution = pruningEnumerator.postprocessing();
        endTime = System.currentTimeMillis();
        System.out.println("Prune-Based Enumerator finished for "+(endTime-startTime)/1000+"s");

        // compare solution
        ResourceTestUtils.assertEqualSolutionPoints(expectedSolution, actualSolution);
    }
}
