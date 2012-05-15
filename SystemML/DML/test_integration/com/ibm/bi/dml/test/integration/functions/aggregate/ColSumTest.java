package com.ibm.bi.dml.test.integration.functions.aggregate;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.LanguageException;


/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 *  <li>general test</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * <ul>
 *  <li>scalar test</li>
 * </ul>
 * 
 * @author schnetter
 */
public class ColSumTest extends AutomatedTestBase {

    private final static String TEST_DIR = "functions/aggregate/";
    private final static String TEST_GENERAL = "General";
    private final static String TEST_SCALAR = "Scalar";


    @Override
    public void setUp() {
        // positive tests
        addTestConfiguration(TEST_GENERAL, new TestConfiguration(TEST_DIR, "ColSumTest", new String[] {
                "vector_colsum", "matrix_colsum" }));

        // negative tests
        addTestConfiguration(TEST_SCALAR, new TestConfiguration(TEST_DIR, "ColSumScalarTest",
                new String[] { "computed" }));
    }

    @Test
    public void testGeneral() {
        int rows = 10;
        int cols = 10;

        TestConfiguration config = getTestConfiguration(TEST_GENERAL);
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);

        loadTestConfiguration(TEST_GENERAL);

        double[][] vector = getRandomMatrix(rows, 1, 0, 1, 1, -1);
        double[][] vectorColSum = new double[1][1];
        for (int i = 0; i < rows; i++) {
            vectorColSum[0][0] += vector[i][0];
        }
        writeInputMatrix("vector", vector);
        writeExpectedMatrix("vector_colsum", vectorColSum);

        double[][] matrix = getRandomMatrix(rows, cols, 0, 1, 1, -1);
        double[][] matrixColSum = new double[1][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixColSum[0][j] += matrix[i][j];
            }
        }
        writeInputMatrix("matrix", matrix);
        writeExpectedMatrix("matrix_colsum", matrixColSum);
disableOutAndExpectedDeletion();
        runTest();

        compareResults(1e-14);
    }

    @Test
    public void testScalar() {
        int scalar = 12;

        TestConfiguration config = getTestConfiguration(TEST_SCALAR);
        config.addVariable("scalar", scalar);

        createHelperMatrix();

        loadTestConfiguration(TEST_SCALAR);

        runTest(true, LanguageException.class);
    }

}
