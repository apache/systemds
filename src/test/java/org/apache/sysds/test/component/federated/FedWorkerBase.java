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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;

public abstract class FedWorkerBase {
	protected static final Log LOG = LogFactory.getLog(FedWorkerBase.class.getName());

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
