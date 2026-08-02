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

package org.apache.sysds.test.component.federated;

import static org.junit.Assert.fail;

import java.net.InetAddress;
import java.net.InetSocketAddress;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;

public abstract class FedWorkerBase {
	protected static final Log LOG = LogFactory.getLog(FedWorkerBase.class.getName());

	/** Upper bound (ms) for {@link #awaitCompressed(long)} polling against async worker-side compression. */
	protected static final int COMPRESS_TIMEOUT_MS = 10_000;

	/** Poll interval used by {@link #awaitCompressed(long)} between successive reads. */
	private static final int COMPRESS_POLL_INTERVAL_MS = 25;

	private final InetSocketAddress addr;
	public final int port;

	public FedWorkerBase(int port) {
		this.port = port;
		addr = getAddr();
	}

	protected static String getConfPath() {
		return "src/test/resources/component/federated/def.xml";
	}

	protected static int startWorker() {
		return startWorker(getConfPath());
	}

	protected static int startWorker(String confPath) {
		final int port = AutomatedTestBase.getRandomAvailablePort();
		AutomatedTestBase.startLocalFedWorkerThread(port, new String[] {"-config", confPath}, 5000);
		return port;
	}

	public long putDouble(double v) {
		return FederatedTestUtils.putDouble(v, addr);
	}

	public double getDouble(long id) {
		return FederatedTestUtils.getDouble(id, addr);
	}

	public long putMatrixBlock(MatrixBlock mb) {
		return FederatedTestUtils.putMatrixBlock(mb, addr);
	}

	public MatrixBlock getMatrixBlock(long id) {
		return FederatedTestUtils.getMatrixBlock(id, addr);
	}

	/**
	 * Poll the federated worker until the matrix at {@code id} is observed as a
	 * {@link CompressedMatrixBlock}, or {@link #COMPRESS_TIMEOUT_MS} elapses.
	 *
	 * <p>Federated workers compress asynchronously after a PUT/READ_VAR (see
	 * {@code CompressedMatrixBlockFactory.compressAsync}), so a {@code getMatrixBlock} fired right
	 * after the operation can race against the in-flight compression and return the uncompressed
	 * block. Tests that need to observe the compressed form should poll instead of sleeping a fixed
	 * amount.
	 *
	 * <p>On timeout this returns the most recent (uncompressed) read so the caller can produce a
	 * meaningful assertion failure naming the variable.
	 *
	 * @param id federated variable id
	 * @return the matrix block, compressed if compression finished in time, otherwise the latest read
	 */
	public MatrixBlock awaitCompressed(long id) {
		final long deadline = System.currentTimeMillis() + COMPRESS_TIMEOUT_MS;
		MatrixBlock mb = getMatrixBlock(id);
		while(!(mb instanceof CompressedMatrixBlock) && System.currentTimeMillis() < deadline) {
			try {
				Thread.sleep(COMPRESS_POLL_INTERVAL_MS);
			}
			catch(InterruptedException ie) {
				Thread.currentThread().interrupt();
				fail("Interrupted while waiting for federated compression of id=" + id);
			}
			mb = getMatrixBlock(id);
		}
		return mb;
	}

	public long matrixMult(long idLeft, long idRight) {
		return FederatedTestUtils.exec_MM(idLeft, idRight, addr);
	}

	public long readMatrix(String path){
		return FederatedTestUtils.readMatrixBlock(path, addr);
	}

	private InetSocketAddress getAddr() {
		try {
			return new InetSocketAddress(InetAddress.getByName("localhost"), port);
		}
		catch(Exception e) {
			fail("Should not happen");
			return null;
		}
	}

}
