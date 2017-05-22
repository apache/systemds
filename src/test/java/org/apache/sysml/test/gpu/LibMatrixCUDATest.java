package org.apache.sysml.test.gpu;

import org.apache.commons.math3.util.FastMath;
import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.MatrixMetadata;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptFactory;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Unit tests for GPU methods
 */
public class LibMatrixCUDATest extends AutomatedTestBase {

    private final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
    private final static String TEST_NAME = "LibMatrixCUDATest";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_DIR, TEST_NAME);
        getAndLoadTestConfiguration(TEST_NAME);
    }

	@Test
	public void test1() throws Exception {
        SparkSession spark = createSystemMLSparkSession("LibMatrixCUDATest", "local");

        MLContext cpuMLC = new MLContext(spark);
        Script generateScript = ScriptFactory.dmlFromString("A = rand(rows=100, cols=100)").out("A");
        Matrix A = cpuMLC.execute(generateScript).getMatrix("A");

        Script sinScript = ScriptFactory.dmlFromString("B = sin(A)").in("A", A).out("B");
        Matrix B = cpuMLC.execute(sinScript).getMatrix("B");
        cpuMLC.close();
        spark.stop();

        MLContext gpuMLC = new MLContext(spark);
        gpuMLC.setGPU(true);
        gpuMLC.setForceGPU(true);
        gpuMLC.setStatistics(true);
        sinScript = ScriptFactory.dmlFromString("B = sin(A)").in("A", A).out("B");
        Matrix BGpu = gpuMLC.execute(sinScript).getMatrix("B");

        double[][] b = B.to2DDoubleArray();
        double[][] bgpu = BGpu.to2DDoubleArray();

        for (int i = 0; i < b.length; i++) {
            Assert.assertArrayEquals(b[i], bgpu[i], 1e-9);
        }
        gpuMLC.close();
        spark.stop();

	}

    @After
    public void tearDown() {
        super.tearDown();
    }

}