package org.apache.sysds.test.functions.federated.paramserv;

import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
public class FederatedParamservTest extends AutomatedTestBase {
    private final static String TEST_DIR = "functions/federated/paramserv/";
    private final static String TEST_NAME = "FederatedParamservTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedParamservTest.class.getSimpleName() + "/";
    private final static int _blocksize = 1024;

    private int _epochs;
    private int _batch_size;
    private double _eta;
    private String _utype;
    private String _freq;

    // parameters


    @Parameterized.Parameters
    public static Collection parameters() {
        return Arrays.asList(new Object[][] {
                {5, 1, 0.01, "BSP", "BATCH"},
                {5, 1, 0.01, "ASP", "BATCH"},
                {5, 1, 0.01, "BSP", "EPOCH"},
                {5, 1, 0.01, "ASP", "EPOCH"},
        });
    }

    public FederatedParamservTest(int epochs, int batch_size, double eta, String utype, String freq) {
        _epochs = epochs;
        _batch_size = batch_size;
        _eta = eta;
        _utype = utype;
        _freq = freq;
    }

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
    }

    @Test
    public void federatedParamserv() {
        // config
        getAndLoadTestConfiguration(TEST_NAME);
        String HOME = SCRIPT_DIR + TEST_DIR;
        setOutputBuffering(true);

        int C = 1, Hin = 28, Win = 28;
        int numFeatures = C*Hin*Win;
        int numLabels = 10;
        int examplesPerWorker = 2;

        // write row partitioned features to disk
        writeInputMatrixWithMTD("X1", generateDummyMNISTFeatures(examplesPerWorker, C, Hin, Win),false,
                        new MatrixCharacteristics(examplesPerWorker, numFeatures, _blocksize, examplesPerWorker * numFeatures));
        writeInputMatrixWithMTD("X2", generateDummyMNISTFeatures(examplesPerWorker, C, Hin, Win),false,
                new MatrixCharacteristics(examplesPerWorker, numFeatures, _blocksize, examplesPerWorker * numFeatures));

        // write row partitioned labels to disk
        writeInputMatrixWithMTD("y1", generateDummyMNISTLabels(examplesPerWorker, numLabels),false,
                new MatrixCharacteristics(examplesPerWorker, numLabels, _blocksize, examplesPerWorker * numLabels));
        writeInputMatrixWithMTD("y2", generateDummyMNISTLabels(examplesPerWorker, numLabels),false,
                new MatrixCharacteristics(examplesPerWorker, numLabels, _blocksize, examplesPerWorker * numLabels));

        // start workers
        int port1 = getRandomAvailablePort();
        int port2 = getRandomAvailablePort();
        Thread thread1 = startLocalFedWorkerThread(port1);
        Thread thread2 = startLocalFedWorkerThread(port2);

        // run test
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] {"-nvargs",
                "X1=" + TestUtils.federatedAddress(port1, input("X1")),
                "X2=" + TestUtils.federatedAddress(port2, input("X2")),
                "y1=" + TestUtils.federatedAddress(port1, input("y1")),
                "y2=" + TestUtils.federatedAddress(port2, input("y2")),
                "examples_per_worker=" + examplesPerWorker,
                "num_features=" + numFeatures,
                "num_labels=" + numLabels,
                "epochs=" + _epochs,
                "batch_size=" + _batch_size,
                "eta=" + _eta,
                "utype=" + _utype,
                "freq=" + _freq
            };

        ByteArrayOutputStream stdout= runTest(null);
        System.out.print(stdout.toString());

        //cleanup
        TestUtils.shutdownThreads(thread1, thread2);
    }

    /**
     * Generates an feature matrix that has the same format as the MNIST dataset,
     * but is completely random and normalized
     *
     *  @param numExamples Number of examples to generate
     *  @param C Channels in the input data
     *  @param Hin Height in Pixels of the input data
     *  @param Win Width in Pixels of the input data
     *  @return a dummy MNIST feature matrix
     */
    private double[][] generateDummyMNISTFeatures(int numExamples, int C, int Hin, int Win) {
        // Seed -1 takes the time in milliseconds as a seed
        // Sparsity 1 means no sparsity
        return getRandomMatrix(numExamples, C*Hin*Win, 0, 1, 1, -1);
    }

    /**
     * Generates an label matrix that has the same format as the MNIST dataset, but is completely random and consists
     * of one hot encoded vectors as rows
     *
     *  @param numExamples Number of examples to generate
     *  @param numLabels Number of labels to generate
     *  @return a dummy MNIST lable matrix
     */
    private double[][] generateDummyMNISTLabels(int numExamples, int numLabels) {
        // Seed -1 takes the time in milliseconds as a seed
        // Sparsity 1 means no sparsity
        return getRandomMatrix(numExamples, numLabels, 0, 1, 1, -1);
    }
}
