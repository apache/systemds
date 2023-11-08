package org.apache.sysds.test.functions.dnn;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;

public class LSTMTest extends AutomatedTestBase {
    String TEST_NAME1 = "LSTMForwardTest";
    String TEST_NAME2 = "LSTMBackwardTest";
    private final static String TEST_DIR = "functions/tensor/";
    private final static String TEST_CLASS_DIR = TEST_DIR + LSTMTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
        addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2));
    }

    @Test
    public void testLSTMForwardLocalSingleSample1(){
        runLSTMTest(1, 32, 1,1, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocalSingleSample2(){
        runLSTMTest(1, 1, 64,1, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocalSingleSample3(){
        runLSTMTest(1, 1, 1,2048, TEST_NAME1);
    }

    //note elias: for large hidden sizes there is discrepancy between built-in and the dml script
    @Test
    public void testLSTMForwardLocalSingleSample4(){
        runLSTMTest(1, 32, 32,1025, 0,0, 1e-2, TEST_NAME1,false);
    }

    @Test
    public void testLSTMForwardLocal1(){
        runLSTMTest(64, 2, 2,2, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocal2(){
        runLSTMTest(32, 8, 1,1, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocal3(){
        runLSTMTest(32, 1, 64,1, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocal4(){
        runLSTMTest(32, 8, 36,1025, TEST_NAME1);
    }

    @Test
    public void testLSTMForwardLocal5(){
        runLSTMTest(32, 75, 128,256, 0, 1, 1e-3, TEST_NAME1, false);
    }

    @Test
    public void testLSTMBackwardLocalSingleSample1(){
        runLSTMTest(1, 2, 3,4,0,1,1e-5, TEST_NAME2, true);
    }

    @Test
    public void testLSTMBackwardLocal1(){
        runLSTMTest(64, 32, 16,32,0,0,1e-5, TEST_NAME2, true);
    }

    @Test
    public void testLSTMBackwardLocal2(){
        runLSTMTest(64, 32, 16,32,0,1,1e-5, TEST_NAME2, true);
    }

    @Test
    @Ignore
    public void testLSTMForwardLocalLarge(){
        runLSTMTest(100, 32, 128,64, 0, 1, 1e-5, TEST_NAME1, false);
    }

    @Test
    @Ignore
    public void testLSTMBackwardLocalLarge(){
        runLSTMTest(128, 128, 128,64, 0, 0, 1e-5, TEST_NAME2, true);
    }

    private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, String testname){
        runLSTMTest(batch_size, seq_length, num_features, hidden_size,0, testname);
    }

    private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, int debug, String testname){
        runLSTMTest(batch_size, seq_length, num_features, hidden_size,debug, 0, 1e-5, testname, false);
    }

    private void runLSTMTest(double batch_size, double seq_length, double num_features, double hidden_size, int debug, int seq,  double precision, String testname, boolean backward)
    {
        //set runtime platform
        Types.ExecMode rtold = setExecMode(Types.ExecMode.SINGLE_NODE);
        try
        {
            getAndLoadTestConfiguration(testname);
            fullDMLScriptName = getScript();

            //run script
            //"-explain", "runtime",
            programArgs = new String[]{"-stats","-args", String.valueOf(batch_size), String.valueOf(seq_length),
                    String.valueOf(num_features), String.valueOf(hidden_size), String.valueOf(debug), String.valueOf(seq),
                    output("1A"),output("1B"),output("2A"), output("2B"),output("3A"),output("3B"),"","","",""};
            int offset = 0;
            if(backward){
                programArgs[14 + offset] = output("4A");
                programArgs[15 + offset] = output("4B");
                programArgs[16 + offset] = output("5A");
                programArgs[17 + offset] = output("5B");
            }
            //output("4A"), output("4B"),output("5A"),output("5B")
            runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

            // Compare results
            extracted(precision,"1");
            extracted(precision,"2");
            extracted(precision,"3");
            if(backward){
                extracted(precision,"4");
                extracted(precision,"5");
            }
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
        finally {
            resetExecMode(rtold);
        }
    }

    private void extracted(double precision, String output) {
        HashMap<MatrixValue.CellIndex, Double> res_actual = readDMLMatrixFromOutputDir(output+"A");
        double[][] resultActualDouble = TestUtils.convertHashMapToDoubleArray(res_actual);
        HashMap<MatrixValue.CellIndex, Double> res_expected = readDMLMatrixFromOutputDir(output+"B");
        double[][] resultExpectedDouble = TestUtils.convertHashMapToDoubleArray(res_expected);
        TestUtils.compareMatrices(resultExpectedDouble, resultActualDouble, precision);
    }
}
