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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
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
public class FederatedParamservTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedParamservTest.class.getName());
	private final static String TEST_DIR = "functions/federated/paramserv/";
	private final static String TEST_NAME = "FederatedParamservTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedParamservTest.class.getSimpleName() + "/";
	private final static int _blocksize = 1024;

	private final String _networkType;
	private final int _numFederatedWorkers;
	private final int _examplesPerWorker;
	private final int _epochs;
	private final int _batch_size;
	private final double _eta;
	private final String _utype;
	private final String _freq;

	// parameters
	@Parameterized.Parameters
	public static Collection<Object[]> parameters() {
		return Arrays.asList(new Object[][] {
			// Network type, number of federated workers, examples per worker, batch size, epochs, learning rate, update
			// type, update frequency
			{"TwoNN", 2, 2, 1, 5, 0.01, "BSP", "BATCH"}, {"TwoNN", 2, 2, 1, 5, 0.01, "ASP", "BATCH"},
			{"TwoNN", 2, 2, 1, 5, 0.01, "BSP", "EPOCH"}, {"TwoNN", 2, 2, 1, 5, 0.01, "ASP", "EPOCH"},
			{"CNN", 2, 2, 1, 5, 0.01, "BSP", "BATCH"}, {"CNN", 2, 2, 1, 5, 0.01, "ASP", "BATCH"},
			{"CNN", 2, 2, 1, 5, 0.01, "BSP", "EPOCH"}, {"CNN", 2, 2, 1, 5, 0.01, "ASP", "EPOCH"},
			{"TwoNN", 5, 1000, 200, 2, 0.01, "BSP", "BATCH"},
			// {"TwoNN", 5, 1000, 200, 2, 0.01, "ASP", "BATCH"},
			// {"TwoNN", 5, 1000, 200, 2, 0.01, "BSP", "EPOCH"},
			// {"TwoNN", 5, 1000, 200, 2, 0.01, "ASP", "EPOCH"},
			// {"CNN", 5, 1000, 200, 2, 0.01, "BSP", "BATCH"},
			// {"CNN", 5, 1000, 200, 2, 0.01, "ASP", "BATCH"},
			{"CNN", 5, 1000, 200, 2, 0.01, "BSP", "EPOCH"},
			// {"CNN", 5, 1000, 200, 2, 0.01, "ASP", "EPOCH"}
		});
	}

	public FederatedParamservTest(String networkType, int numFederatedWorkers, int examplesPerWorker, int batch_size,
		int epochs, double eta, String utype, String freq) {
		_networkType = networkType;
		_numFederatedWorkers = numFederatedWorkers;
		_examplesPerWorker = examplesPerWorker;
		_batch_size = batch_size;
		_epochs = epochs;
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
	public void federatedParamservSingleNode() {
		federatedParamserv(ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedParamservHybrid() {
		federatedParamserv(ExecMode.HYBRID);
	}

	private void federatedParamserv(ExecMode mode) {
		// config
		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;
		setOutputBuffering(true);

		int C = 1, Hin = 28, Win = 28;
		int numFeatures = C * Hin * Win;
		int numLabels = 10;

		ExecMode platformOld = setExecMode(mode);

		try {

			// dml name
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			// generate program args
			List<String> programArgsList = new ArrayList<>(Arrays.asList("-stats",
				"-nvargs",
				"examples_per_worker=" + _examplesPerWorker,
				"num_features=" + numFeatures,
				"num_labels=" + numLabels,
				"epochs=" + _epochs,
				"batch_size=" + _batch_size,
				"eta=" + _eta,
				"utype=" + _utype,
				"freq=" + _freq,
				"network_type=" + _networkType,
				"channels=" + C,
				"hin=" + Hin,
				"win=" + Win));

			// for each worker
			List<Integer> ports = new ArrayList<>();
			List<Thread> threads = new ArrayList<>();
			for(int i = 0; i < _numFederatedWorkers; i++) {
				// write row partitioned features to disk
				writeInputMatrixWithMTD("X" + i,
					generateDummyMNISTFeatures(_examplesPerWorker, C, Hin, Win),
					false,
					new MatrixCharacteristics(_examplesPerWorker, numFeatures, _blocksize,
						_examplesPerWorker * numFeatures));
				// write row partitioned labels to disk
				writeInputMatrixWithMTD("y" + i,
					generateDummyMNISTLabels(_examplesPerWorker, numLabels),
					false,
					new MatrixCharacteristics(_examplesPerWorker, numLabels, _blocksize,
						_examplesPerWorker * numLabels));

				// start worker
				ports.add(getRandomAvailablePort());
				threads.add(startLocalFedWorkerThread(ports.get(i), FED_WORKER_WAIT_S));

				// add worker to program args
				programArgsList.add("X" + i + "=" + TestUtils.federatedAddress(ports.get(i), input("X" + i)));
				programArgsList.add("y" + i + "=" + TestUtils.federatedAddress(ports.get(i), input("y" + i)));
			}
			try {
				Thread.sleep(1000);
			}
			catch(InterruptedException e) {
				e.printStackTrace();
			}
	
			programArgs = programArgsList.toArray(new String[0]);
			LOG.debug(runTest(null));
			Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
			
			// cleanup
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
