package org.apache.sysds.test.functions.federated.paramserv;

import com.sun.xml.internal.messaging.saaj.util.ByteOutputStream;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.ByteArrayOutputStream;

public class FederatedParamservTest extends AutomatedTestBase {
    private final static String TEST_DIR = "functions/federated/paramserv/";
    private final static String TEST_NAME = "FederatedParamservTest";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedParamservTest.class.getSimpleName() + "/";
    private final static int blocksize = 1024;

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
        int examplesPerWorker = 10;

        // write row partitioned features to disk
        writeInputMatrixWithMTD("X1", generateDummyMNISTFeatures(examplesPerWorker, C, Hin, Win),false,
                        new MatrixCharacteristics(examplesPerWorker, numFeatures, blocksize, examplesPerWorker * numFeatures));
        writeInputMatrixWithMTD("X2", generateDummyMNISTFeatures(examplesPerWorker, C, Hin, Win),false,
                new MatrixCharacteristics(examplesPerWorker, numFeatures, blocksize, examplesPerWorker * numFeatures));

        // write row partitioned labels to disk
        writeInputMatrixWithMTD("y1", generateDummyMNISTLabels(examplesPerWorker, numLabels),false,
                new MatrixCharacteristics(examplesPerWorker, numLabels, blocksize, examplesPerWorker * numLabels));
        writeInputMatrixWithMTD("y2", generateDummyMNISTLabels(examplesPerWorker, numLabels),false,
                new MatrixCharacteristics(examplesPerWorker, numLabels, blocksize, examplesPerWorker * numLabels));

        // start workers
        int port1 = getRandomAvailablePort();
        int port2 = getRandomAvailablePort();
        Process process1 = startLocalFedWorker(port1);
        Process process2 = startLocalFedWorker(port2);

        // run test
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] {"-nvargs",
                "X1=" + TestUtils.federatedAddress(port1, input("X1")),
                "X2=" + TestUtils.federatedAddress(port2, input("X2")),
                "y1=" + TestUtils.federatedAddress(port1, input("y1")),
                "y2=" + TestUtils.federatedAddress(port2, input("y2")),
                "examples_per_worker=" + examplesPerWorker,
                "num_features=" + numFeatures,
                "num_labels=" + numLabels
                };
        ByteArrayOutputStream stdout= runTest(null);
        System.out.print(stdout.toString());

        //cleanup
        TestUtils.shutdownThreads(process1, process2);
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
