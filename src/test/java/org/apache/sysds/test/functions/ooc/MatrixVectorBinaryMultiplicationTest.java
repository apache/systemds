package org.apache.sysds.test.functions.ooc;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.functions.binary.matrix.MatrixVectorTest;
import org.junit.Test;

import java.io.IOException;

public class MatrixVectorBinaryMultiplicationTest extends AutomatedTestBase {
    private final static String TEST_NAME1 = "MatrixVectorMultiplication";
    private final static String TEST_DIR = "functions/ooc/";
    private final static String TEST_CLASS_DIR = TEST_DIR + MatrixVectorTest.class.getSimpleName() + "/";
    private final static double eps = 1e-10;

    private final static int rows = 3500;
    private final static int cols_wide = 1500;
    private final static int cols_skinny = 500;

    private final static double sparsity1 = 0.7;
    private final static double sparsity2 = 0.1;

    @Override
    public void setUp()
    {
        addTestConfiguration(TEST_NAME1,
                new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
    }

    @Test
    public void testMVBinaryMultiplication()
    {
        runMatrixVectorMultiplicationTest(cols_wide, false);
    }


    private void runMatrixVectorMultiplicationTest(int cols, boolean sparse )
    {

        Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

        try
        {
            getAndLoadTestConfiguration(TEST_NAME1);

            /* This is for running the junit test the new way, i.e., construct the arguments directly */
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
            programArgs = new String[]{"-explain", "-stats", "-ooc",
                    "-args", input("A"), input("x"), output("y")};

            fullRScriptName = HOME + TEST_NAME1 + ".R";
            rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

            //generate actual dataset
//            double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 10);
//            MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, -1, -1);
//            writeInputMatrixWithMTD("A", A, true, mc);
//            double[][] x = getRandomMatrix(cols, 1, 0, 1, 1.0, 10);
//            mc = new MatrixCharacteristics(cols, 1, -1, cols);
//            writeInputMatrixWithMTD("x", x, true, mc);
            // 1. Generate the data in-memory as MatrixBlock objects
            double[][] A_data = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 10);
            double[][] x_data = getRandomMatrix(cols, 1, 0, 1, 1.0, 10);

            // 2. Convert the double arrays to MatrixBlock objects
            MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);
            MatrixBlock x_mb = DataConverter.convertToMatrixBlock(x_data);

            // 3. Create a binary matrix writer
            MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

            // 4. Write matrix A to a binary SequenceFile
            writer.writeMatrixToHDFS(A_mb, input("A"), rows, cols, 1000, A_mb.getNonZeros());
            HDFSTool.writeMetaDataFile(input("A.mtd"), Types.ValueType.FP64,
                    new MatrixCharacteristics(rows, cols, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);

            // 5. Write vector x to a binary SequenceFile
            writer.writeMatrixToHDFS(x_mb, input("x"), cols, 1, 1000, x_mb.getNonZeros());
            HDFSTool.writeMetaDataFile(input("x.mtd"), Types.ValueType.FP64,
                    new MatrixCharacteristics(cols, 1, 1000, x_mb.getNonZeros()), Types.FileFormat.BINARY);


            boolean exceptionExpected = false;
            runTest(true, exceptionExpected, null, -1);
//            runRScript(true);
//            compareResultsWithR(eps);

        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            resetExecMode(platformOld);
        }

    }

    private static double[][] readMatrix(String fname, Types.FileFormat fmt, long rows, long cols, int brows, int bcols )
            throws IOException
    {
        MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, brows, bcols);
        double[][] C = DataConverter.convertToDoubleMatrix(mb);
        return C;
    }
}
