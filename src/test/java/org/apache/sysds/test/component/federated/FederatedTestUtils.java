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

import java.net.InetSocketAddress;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class FederatedTestUtils {
	protected static final Log LOG = LogFactory.getLog(FederatedTestUtils.class.getName());

	public static long putDouble(double v, InetSocketAddress addr) {
		return putDouble(v, addr, 5000);
	}

	public static long putDouble(double v, InetSocketAddress addr, int timeout) {
		try {
			final ScalarObject sb = new DoubleObject(v);
			final long id = FederationUtils.getNextFedDataID();
			final FederatedRequest frq = new FederatedRequest(RequestType.PUT_VAR, null, id, sb);
			final FederatedResponse r = FederatedData.executeFederatedOperation(addr, frq).get(timeout,
				TimeUnit.MILLISECONDS);
			if(r.isSuccessful())
				return id;
			else
				fail("Failed putting scalar into worker");
		}
		catch(TimeoutException e) {
			fail("Failed to put scaler within timeout.");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from put scalar");
		}
		return -1;
	}

	public static double getDouble(long id, InetSocketAddress addr) {
		return getDouble(id, addr, 5000);
	}

	public static double getDouble(long id, InetSocketAddress addr, int timeout) {
		try {
			FederatedRequest frq = new FederatedRequest(RequestType.GET_VAR, id);
			FederatedResponse r = FederatedData.executeFederatedOperation(addr, frq).get(timeout, TimeUnit.MILLISECONDS);
			ScalarObject sb = (ScalarObject) r.getData()[0];
			return sb.getDoubleValue();
		}
		catch(TimeoutException e) {
			fail("Failed to put matrix block within timeout.");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from put Matrix Block");
		}
		return Double.NaN;
	}

	public static long putMatrixBlock(MatrixBlock mb, InetSocketAddress addr) {
		return putMatrixBlock(mb, addr, 5000);
	}

	public static long putMatrixBlock(MatrixBlock mb, InetSocketAddress addr, int timeout) {
		try {
			final long id = FederationUtils.getNextFedDataID();
			final FederatedRequest frq = new FederatedRequest(RequestType.PUT_VAR, null, id, mb);
			final Future<FederatedResponse> fr = FederatedData.executeFederatedOperation(addr, frq);
			final FederatedResponse r = fr.get(timeout, TimeUnit.MILLISECONDS);
			LOG.error(r);
			if(r.isSuccessful())
				return id;
			else
				fail("Failed putting matrix block into worker");
		}
		catch(TimeoutException e) {
			fail("Failed to put matrix block within timeout.");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from put Matrix Block");
		}
		// Should never hit here.
		return -1;
	}

	public static MatrixBlock getMatrixBlock(long id, InetSocketAddress addr) {
		return getMatrixBlock(id, addr, 5000);
	}

	public static MatrixBlock getMatrixBlock(long id, InetSocketAddress addr, int timeout) {
		try {
			FederatedRequest frq = new FederatedRequest(RequestType.GET_VAR, id);
			Future<FederatedResponse> fr = FederatedData.executeFederatedOperation(addr, frq);
			FederatedResponse r = fr.get(timeout, TimeUnit.MILLISECONDS);
			return (MatrixBlock) r.getData()[0];
		}
		catch(TimeoutException e) {
			fail("Failed to put matrix block within timeout.");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from put Matrix Block");
		}
		return null;
	}

	public static long readMatrixBlock(String path, InetSocketAddress addr){
		return readMatrixBlock(path, addr, 5000);
	}

	public static long readMatrixBlock(String path, InetSocketAddress addr, int timeout) {
		try {
			final long id = FederationUtils.getNextFedDataID();
			final FederatedRequest frq = new FederatedRequest(RequestType.READ_VAR, id, new Object[]{"./"+path, DataType.MATRIX.toString()});
			final Future<FederatedResponse> fr = FederatedData.executeFederatedOperation(addr, frq);
			final FederatedResponse r = fr.get(timeout, TimeUnit.MILLISECONDS);
			if(r.isSuccessful())
				return id;
			else
				fail("Failed reading matrix block into worker");
		}
		catch(TimeoutException e) {
			fail("Failed reading matrix block within timeout.");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from reading Matrix Block");
		}
		// Should never hit here.
		return -1;
	}

	public static long exec_MM(long idLeft, long idRight, InetSocketAddress addr) {
		return exec_MM(idLeft, idRight, addr, 5000);
	}

	public static long exec_MM(long idLeft, long idRight, InetSocketAddress addr, int timeout) {
		final long idOut = FederationUtils.getNextFedDataID();
		String inst = InstructionUtils.concatOperands("CP", "ba+*", Long.toString(idLeft), Long.toString(idRight),
			Long.toString(idOut), Integer.toString(16));
		exec(idOut, inst, addr, timeout);
		return idOut;
	}

	private static void exec(long id, String inst, InetSocketAddress addr, int timeout) {
		try {
			final FederatedRequest frq = new FederatedRequest(RequestType.EXEC_INST, id, inst);
			final Future<FederatedResponse> fr = FederatedData.executeFederatedOperation(addr, frq);
			final FederatedResponse r = fr.get(timeout, TimeUnit.MILLISECONDS);
			if(!r.isSuccessful())
				fail("Failed to execute instruction: " + inst);
		}
		catch(TimeoutException e) {
			fail("Failed to execute instruction within timeout: " + inst);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to get response from put Matrix Block");
		}
	}

	protected static void wait(int ms) {
		try {
			Thread.sleep(ms);
		}
		catch(Exception e) {
			fail("Failed to wait");
		}
	}
}
