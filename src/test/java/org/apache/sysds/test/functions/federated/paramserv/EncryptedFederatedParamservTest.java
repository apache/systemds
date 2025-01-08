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
import java.util.Objects;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.paramserv.NativeHEHelper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.fail;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class EncryptedFederatedParamservTest extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(EncryptedFederatedParamservTest.class.getName());
	private final static String TEST_DIR = "functions/federated/paramserv/";
	private final static String TEST_NAME = "EncryptedFederatedParamservTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + EncryptedFederatedParamservTest.class.getSimpleName() + "/";

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

	// parameters
	@Parameterized.Parameters
	public static Collection<Object[]> parameters() {
		return Arrays.asList(new Object[][] {
				// Network type, number of federated workers, data set size, batch size, epochs, learning rate, update type, update frequency
				// basic functionality
				//{"TwoNN",	4, 60000, 32, 4, 0.01, 	"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"NONE" ,		"false","BALANCED",		200},

				// One important point is that we do the model averaging in the case of BSP
				{"UNet",	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER",	"BASELINE",		"false",	"BALANCED",		200},
				//{"UNet",	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"BASELINE",		"false",	"IMBALANCED",	200},
				{"TwoNN",	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"BASELINE",		"false",	"IMBALANCED",	200},
				{"CNN", 	2, 4, 1, 1, 0.01, 		"BSP", "EPOCH", "KEEP_DATA_ON_WORKER",  "BASELINE",		"false",	"IMBALANCED", 	200},
				//{"TwoNN", 	5, 1000, 100, 1, 0.01, 	"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"NONE",			"true",	"BALANCED",		200},
				{"TwoNN",	2, 4, 1, 4, 0.01, 		"SBP", "BATCH", "KEEP_DATA_ON_WORKER", 	"BASELINE",		"false",	"IMBALANCED",	200},
				{"TwoNN",	2, 4, 1, 4, 0.01, 		"SBP", "BATCH", "KEEP_DATA_ON_WORKER", 	"BASELINE",		"false",	"BALANCED",		200},
				//{"CNN",		2, 4, 1, 4, 0.01, 		"SBP", "EPOCH", "SHUFFLE",			 	"BASELINE",		"false",	"BALANCED",		200},

				/*
					// runtime balancing
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_MIN", 	"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "EPOCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_MIN", 	"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_AVG", 	"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "EPOCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_AVG", 	"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "BATCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_MAX",	"true", "IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 4, 0.01, 		"BSP", "EPOCH", "KEEP_DATA_ON_WORKER", 	"CYCLE_MAX",	"true", "IMBALANCED",	200},
		
					// data partitioning
					{"TwoNN", 	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "SHUFFLE", 				"CYCLE_AVG", 	"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "REPLICATE_TO_MAX",	 	"NONE", 		"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "SUBSAMPLE_TO_MIN",		"NONE", 		"true",	"IMBALANCED",	200},
					{"TwoNN", 	2, 4, 1, 1, 0.01, 		"BSP", "BATCH", "BALANCE_TO_AVG",		"NONE", 		"true",	"IMBALANCED",	200},
		
					// balanced tests
					{"CNN", 	5, 1000, 100, 2, 0.01, 	"BSP", "EPOCH", "KEEP_DATA_ON_WORKER", 	"NONE", 		"true",	"BALANCED",		200}
				*/
		});
	}

	public EncryptedFederatedParamservTest(String networkType, int numFederatedWorkers,
		int dataSetSize, int batch_size, int epochs, double eta, String utype, String freq,
		String scheme, String runtime_balancing, String weighting, String data_distribution, int seed)
	{
		NativeHEHelper.initialize();
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
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void EncryptedfederatedParamservSingleNode() {
		EncryptedfederatedParamserv(ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void EncryptedfederatedParamservHybrid() {
		EncryptedfederatedParamserv(ExecMode.HYBRID, true);
	}

	private void EncryptedfederatedParamserv(ExecMode mode, boolean modelAvg) {
		// Warning Statistics accumulate in unit test
		// config
		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;
		setOutputBuffering(true);

		int C = 1, Hin = 28, Win = 28;
		int numLabels = 10;
		if (Objects.equals(_networkType, "UNet")){
			C = 3; Hin = 196; Win = 196;
			numLabels = C * Hin * Win;
		}

		ExecMode platformOld = setExecMode(mode);
		// start threads
		List<Integer> ports = new ArrayList<>();
		List<Thread> threads = new ArrayList<>();
		try {
			for(int i = 0; i < _numFederatedWorkers; i++) {
				int port = getRandomAvailablePort();
				threads.add(startLocalFedWorkerThread(port,
						i==(_numFederatedWorkers-1) ? FED_WORKER_WAIT : FED_WORKER_WAIT_S));
				ports.add(port);

				if ( threads.get(i).isInterrupted() || !threads.get(i).isAlive() )
					throw new DMLRuntimeException("Federated worker thread dead or interrupted! Port " + port);
			}

			// generate test data
			double[][] features = ParamServTestUtils.generateFeatures(_networkType, _dataSetSize, C, Hin, Win);
			double[][] labels = ParamServTestUtils.generateLabels(_networkType, _dataSetSize, numLabels, C*Hin*Win, features);
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

			//wait for all workers to be setup
			Thread.sleep(FED_WORKER_WAIT);

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
					"modelAvg=" +  Boolean.toString(modelAvg).toUpperCase()));

			programArgs = programArgsList.toArray(new String[0]);
			String log = runTest(null).toString();

			if (!heavyHittersContainsAllString("paramserv"))
				fail("The following expected heavy hitters are missing: "
					+ Arrays.toString(missingHeavyHitters("paramserv")));
			Assert.assertEquals("Test Failed \n" + log, 0, Statistics.getNoOfExecutedSPInst());
		}
		catch(InterruptedException e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {
			// shut down threads
			for ( Thread thread : threads ){
				TestUtils.shutdownThreads(thread);
			}

			resetExecMode(platformOld);
		}
	}
}
