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
package org.apache.sysds.test.functions.federated.io;

import java.io.File;
import java.net.InetSocketAddress;
import java.security.cert.CertificateException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.SSLException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedSSLUtil;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.federated.FederatedTestObjectConstructor;
import org.junit.Assert;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class FederatedSSLTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedSSLTest.class.getName());

	// These tests use the same scripts as the Federated Reader tests, just with SSL enabled.
	private final static String TEST_DIR = "functions/federated/io/";
	private final static String TEST_NAME = "FederatedReaderTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedSSLTest.class.getSimpleName() + "/";
	private final static int blocksize = 1024;
	private final static int rows = 10;
	private final static int cols = 13;

	private final static String CONF_DIR = SCRIPT_DIR + TEST_DIR + "config/";
	// Certificate issued for localhost and signed by the authority the coordinator trusts
	private final static File SIGNED_CONF = new File(CONF_DIR, "SignedSSLConfig.xml");
	// Coordinator trusting an authority unrelated to the one that signed the worker certificate
	private final static File UNTRUSTED_CONF = new File(CONF_DIR, "UntrustedSSLConfig.xml");
	// Certificate signed by the trusted authority, but issued for another host
	private final static File OTHER_HOST_CONF = new File(CONF_DIR, "OtherHostSSLConfig.xml");
	// SSL enabled without configuring any certificate, which is not supported
	private final static File NO_CERT_CONF = new File(SCRIPT_DIR + TEST_DIR + "SSLConfig.xml");

	private File confFile = SIGNED_CONF;
	private File workerConfFile = null;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
		// The SSL context of the coordinator is cached for the JVM, so it should be cleared in between the tests
		FederatedSSLUtil.resetClientContext();
		FederatedData.resetFederatedSites();
	}

	// A certificate signed by the trusted authority and issued for the contacted host is accepted.
	@Test
	public void federatedSinglenodeReadSigned() {
		// the workers and the coordinator share one configuration, the coordinator trusts the signing authority
		confFile = SIGNED_CONF;
		federatedRead(Types.ExecMode.SINGLE_NODE);
	}

	// A certificate signed by an authority the coordinator does not know is rejected.
	@Test
	public void untrustedCertificateIsRejected() {
		workerConfFile = SIGNED_CONF;
		confFile = UNTRUSTED_CONF;
		assertRequestFails();
	}

	// A certificate signed by the trusted authority, but issued for another host, is rejected as well.
	@Test
	public void certificateOfOtherHostIsRejected() {
		confFile = OTHER_HOST_CONF;
		assertRequestFails();
	}

	// A worker cannot serve SSL without a certificate, generating one is not supported.
	@Test
	public void workerCertificateIsRequired() throws Exception {
		confFile = NO_CERT_CONF;
		getAndLoadTestConfiguration(TEST_NAME);
		ConfigurationManager.setGlobalConfig(new DMLConfig(getCurConfigFile().getPath()));

		try {
			FederatedSSLUtil.createServerContext();
			Assert.fail("A worker without a configured certificate should not start SSL.");
		}
		catch(DMLRuntimeException e) {
			Assert.assertTrue("Expected a hint at the certificate configuration but got: " + e.getMessage(),
				e.getMessage().contains(DMLConfig.FEDERATED_SSL_CERT));
		}
	}

	// A coordinator cannot connect without knowing which certificates to trust.
	@Test
	public void trustedCertificatesAreRequired() {
		workerConfFile = SIGNED_CONF;
		confFile = NO_CERT_CONF;
		getAndLoadTestConfiguration(TEST_NAME);
		fullDMLScriptName = "";
		final int port = getRandomAvailablePort();
		final Thread[] workers = startWorkers(port);

		try {
			// the worker startup sets the global configuration in this JVM, therefore the coordinator
			// configuration has to be applied after the workers are up.
			ConfigurationManager.setGlobalConfig(new DMLConfig(getCurConfigFile().getPath()));

			FederatedData.executeFederatedOperation(new InetSocketAddress("localhost", port),
				new FederatedRequest(RequestType.CLEAR)).get(FED_WORKER_WAIT, TimeUnit.MILLISECONDS);
			Assert.fail("A coordinator without trusted certificates should not connect.");
		}
		catch(Exception e) {
			Assert.assertTrue("Expected a hint at the trust configuration but got: " + e,
				messageContains(e, DMLConfig.FEDERATED_SSL_TRUST));
		}
		finally {
			TestUtils.shutdownThreads(workers);
		}
	}

	// Read a federated matrix over SSL and compare against the same matrix read locally.
	private void federatedRead(Types.ExecMode execMode) {
		final Types.ExecMode oldPlatform = setExecMode(execMode);
		getAndLoadTestConfiguration(TEST_NAME);
		setOutputBuffering(true);

		// write input matrices
		final int halfRows = rows / 2;
		final long[][] begins = new long[][] {new long[] {0, 0}, new long[] {halfRows, 0}};
		final long[][] ends = new long[][] {new long[] {halfRows, cols}, new long[] {rows, cols}};
		// We have two matrices handled by a single federated worker
		final double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		final double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		final int port1 = getRandomAvailablePort();
		final int port2 = getRandomAvailablePort();
		final Thread[] workers = startWorkers(port1, port2);

		try {
			final MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(rows, cols, blocksize,
				"localhost", begins, ends, new int[] {port1, port2}, new String[] {input("X1"), input("X2")},
				input("X.json"));
			// FIXME: reset avoids deadlock on reference script
			// (because federated matrix creation added to federated sites - blocks on clear)
			FederatedData.resetFederatedSites();
			writeInputFederatedWithMTD("X.json", fed);

			// Run reference dml script with normal matrix
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "Row2Reference.dml";
			programArgs = new String[] {"-stats", "-args", input("X1"), input("X2")};
			final String refOut = runTest(null).toString();

			// Run federated
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-args", input("X.json")};
			final String out = runTest(null).toString();

			Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
			// Verify output
			Assert.assertEquals(Double.parseDouble(refOut.split("\n")[0]), Double.parseDouble(out.split("\n")[0]),
				0.00001);
		}
		catch(Exception e) {
			e.printStackTrace();
			Assert.fail("Federated read over SSL failed: " + e.getMessage());
		}
		finally {
			resetExecMode(oldPlatform);
			TestUtils.shutdownThreads(workers);
		}
	}

	// The coordinator rejects the certificate of the worker instead of getting an answer to its request.
	private void assertRequestFails() {
		getAndLoadTestConfiguration(TEST_NAME);
		fullDMLScriptName = "";
		final int port = getRandomAvailablePort();
		final Thread[] workers = startWorkers(port);

		try {
			// the worker startup sets the global configuration in this JVM, therefore the coordinator
			// configuration has to be applied after the workers are up.
			ConfigurationManager.setGlobalConfig(new DMLConfig(getCurConfigFile().getPath()));

			final Future<FederatedResponse> f = FederatedData.executeFederatedOperation(
				new InetSocketAddress("localhost", port), new FederatedRequest(RequestType.CLEAR));

			f.get(FED_WORKER_WAIT, TimeUnit.MILLISECONDS);
			Assert.fail("The request to a worker with a rejected certificate should not succeed.");
		}
		catch(ExecutionException e) {
			Assert.assertTrue("Expected an SSL failure but got: " + e.getCause(), isSSLFailure(e.getCause()));
		}
		catch(Exception e) {
			e.printStackTrace();
			Assert.fail("Expected an SSL failure but got: " + e);
		}
		finally {
			TestUtils.shutdownThreads(workers);
		}
	}

	// The workers have to be started with a configuration as well, otherwise they do not enable SSL.
	private Thread[] startWorkers(int... ports) {
		final File conf = (workerConfFile != null) ? workerConfFile : getCurConfigFile();
		return startLocalFedWorkerThreads(ports, new String[] {"-config", conf.getPath()}, FED_WORKER_WAIT);
	}

	private static boolean isSSLFailure(Throwable t) {
		for(Throwable c = t; c != null; c = c.getCause())
			if(c instanceof SSLException || c instanceof CertificateException)
				return true;
		return false;
	}

	private static boolean messageContains(Throwable t, String text) {
		for(Throwable c = t; c != null; c = c.getCause())
			if(c.getMessage() != null && c.getMessage().contains(text))
				return true;
		return false;
	}

	@Override
	protected File getConfigTemplateFile() {
		return confFile;
	}
}
