/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.federated.paramserv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NbatchesFederatedParamservTest extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(NbatchesFederatedParamservTest.class.getName());
	private final static String TEST_DIR = "functions/federated/paramserv/";
	private final static String TEST_NAME = "NbatchesFederatedParamservTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + NbatchesFederatedParamservTest.class.getSimpleName() + "/";

	private final String _networkType;
	private final int _numFederatedWorkers;
	private final int _dataSetSize;
	private final int _epochs;
	private final int _batch_size;
	private final double _eta;
	private final String _utype;
	private final String _freq;
	private final String _scheme;
	private final String _runtime_balancing;
	private final String _weighting;
	private final String _data_distribution;
	private final int _seed;
	private final int _nbatches;

	// parameters
	@Parameterized.Parameters
	public static Collection<Object[]> parameters() {
		return Arrays.asList(new Object[][] {
			// Network type, number of federated workers, data set size, batch size, epochs, learning rate, update type, update frequency, number of batches per nbatches
			// basic functionality
			{"CNN",   2, 2000, 100, 4, 0.01, "BSP", "NBATCHES", "SHUFFLE",             "NONE",      "true", "BALANCED", 200, 8},
			{"CNN",   2, 480,  32,  4, 0.01, "ASP", "NBATCHES", "REPLICATE_TO_MAX",    "CYCLE_MIN", "true", "BALANCED", 200, 16},
			{"TwoNN", 5, 2000, 100, 2, 0.01, "BSP", "NBATCHES", "KEEP_DATA_ON_WORKER", "NONE",      "true", "BALANCED", 200, 2},
			{"CNN",   2, 2000, 100, 4, 0.01, "SBP", "NBATCHES", "SHUFFLE",             "NONE",      "true", "BALANCED", 200, 8},
			{"TwoNN", 5, 2000, 100, 2, 0.01, "BSP", "NBATCHES", "KEEP_DATA_ON_WORKER", "NONE",      "true", "BALANCED", 200, 2},
		});
	}

	public NbatchesFederatedParamservTest(String networkType, int numFederatedWorkers, int dataSetSize, int batch_size,
		int epochs, double eta, String utype, String freq, String scheme, String runtime_balancing, String weighting, String data_distribution, int seed, int nbatches) {

		_networkType = networkType;
		_numFederatedWorkers = numFederatedWorkers;
		_dataSetSize = dataSetSize;
		_batch_size = batch_size;
		_epochs = epochs;
		_eta = eta;
		_utype = utype;
		_freq = freq;
		_scheme = scheme;
		_runtime_balancing = runtime_balancing;
		_weighting = weighting;
		_data_distribution = data_distribution;
		_seed = seed;
		_nbatches = nbatches;
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void federatedParamservSingleNodeWithNBatches() {
		runFederatedParamservTest(ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedParamservHybridWithNBatches() {
		runFederatedParamservTest(ExecMode.HYBRID);
	}

	private void runFederatedParamservTest(ExecMode mode) {
		// Warning Statistics accumulate in unit test
		// config
		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;
		setOutputBuffering(true);

		int C = 1, Hin = 28, Win = 28;
		int numLabels = 10;

		ExecMode platformOld = setExecMode(mode);

		try {
			// start threads
			List<Integer> ports = new ArrayList<>();
			List<Thread> threads = new ArrayList<>();
			for(int i = 0; i < _numFederatedWorkers; i++) {
				ports.add(getRandomAvailablePort());
				threads.add(startLocalFedWorkerThread(ports.get(i), 
					(i==(_numFederatedWorkers-1) ? FED_WORKER_WAIT : FED_WORKER_WAIT_S)));
			}

			// generate test data
			double[][] features = generateDummyMNISTFeatures(_dataSetSize, C, Hin, Win);
			double[][] labels = generateDummyMNISTLabels(_dataSetSize, numLabels);
			String featuresName = "";
			String labelsName = "";

			// federate test data balanced or imbalanced
			if(_data_distribution.equals("IMBALANCED")) {
				featuresName = "X_IMBALANCED_" + _numFederatedWorkers;
				labelsName = "y_IMBALANCED_" + _numFederatedWorkers;
				double[][] ranges = {{0,1}, {1,4}};
				rowFederateLocallyAndWriteInputMatrixWithMTD(featuresName, features, _numFederatedWorkers, ports, ranges);
				rowFederateLocallyAndWriteInputMatrixWithMTD(labelsName, labels, _numFederatedWorkers, ports, ranges);
			}
			else {
				featuresName = "X_BALANCED_" + _numFederatedWorkers;
				labelsName = "y_BALANCED_" + _numFederatedWorkers;
				double[][] ranges = generateBalancedFederatedRowRanges(_numFederatedWorkers, features.length);
				rowFederateLocallyAndWriteInputMatrixWithMTD(featuresName, features, _numFederatedWorkers, ports, ranges);
				rowFederateLocallyAndWriteInputMatrixWithMTD(labelsName, labels, _numFederatedWorkers, ports, ranges);
			}

			try {
				//wait for all workers to be setup
				Thread.sleep(FED_WORKER_WAIT);
			}
			catch(InterruptedException e) {
				e.printStackTrace();
			}

			// dml name
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			// generate program args
			List<String> programArgsList = new ArrayList<>(Arrays.asList("-stats",
				"-nvargs",
				"features=" + input(featuresName),
				"labels=" + input(labelsName),
				"epochs=" + _epochs,
				"batch_size=" + _batch_size,
				"eta=" + _eta,
				"utype=" + _utype,
				"freq=" + _freq,
				"scheme=" + _scheme,
				"runtime_balancing=" + _runtime_balancing,
				"weighting=" + _weighting,
				"network_type=" + _networkType,
				"channels=" + C,
				"hin=" + Hin,
				"win=" + Win,
				"seed=" + _seed,
				"nbatches=" + _nbatches));

			programArgs = programArgsList.toArray(new String[0]);
			String log = runTest(null).toString();
			Assert.assertEquals("Test Failed \n" + log,0, Statistics.getNoOfExecutedSPInst());

			// shut down threads
			for(int i = 0; i < _numFederatedWorkers; i++) {
				TestUtils.shutdownThreads(threads.get(i));
			}
		}
		finally {
			resetExecMode(platformOld);
		}
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
