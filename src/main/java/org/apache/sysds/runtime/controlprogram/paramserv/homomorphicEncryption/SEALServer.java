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

package org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.NativeHEHelper;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.instructions.cp.CiphertextMatrix;
import org.apache.sysds.runtime.instructions.cp.Encrypted;
import org.apache.sysds.runtime.instructions.cp.PlaintextMatrix;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import java.util.Arrays;

public class SEALServer {
	public SEALServer() {
		// TODO take params here, like slot_count etc.
		ctx = NativeHEHelper.initServer();
	}

	// this is a pointer to the context used by all native methods of this class
	private final long ctx;
	private byte[] _a;

	/**
	 * this generates the a constant. in a future version we want to generate this together with the clients to prevent misuse
	 * @return serialized a constant
	 */
	public synchronized byte[] generateA() {
		if (_a == null) {
			_a = NativeHEHelper.generateA(ctx);
		}
		return _a;
	}

	/**
	 * accumulates the given partial public keys into a public key, stores it in ctx and returns it
	 * @param partial_public_keys an array of partial public keys generated with SEALServer::generatePartialPublicKey
	 * @return the aggregated public key
	 */
	public PublicKey aggregatePartialPublicKeys(PublicKey[] partial_public_keys) {
		return new PublicKey(NativeHEHelper.aggregatePartialPublicKeys(ctx, extractRawData(partial_public_keys)));
	}

	/**
	 * accumulates the given ciphertext blocks into a sum ciphertext and returns it
	 * @param ciphertexts ciphertexts encrypted with the partial public keys
	 * @return the accumulated ciphertext (which is the homomorphic sum of ciphertexts)
	 */
	public CiphertextMatrix accumulateCiphertexts(CiphertextMatrix[] ciphertexts) {
		return new CiphertextMatrix(ciphertexts[0].getDims(), ciphertexts[0].getDataCharacteristics(), NativeHEHelper.accumulateCiphertexts(ctx, extractRawData(ciphertexts)));
	}

	/**
	 * averages the partial decryptions
	 * @param encrypted_sum is the result of accumulateCiphertexts()
	 * @param partial_plaintexts is the result of SEALServer::partiallyDecrypt of each ciphertext fed into accumulateCiphertexts
	 * @return the unencrypted, element-wise average of the original matrices
	 */
	public MatrixObject average(CiphertextMatrix encrypted_sum, PlaintextMatrix[] partial_plaintexts) {
		double[] raw_result = NativeHEHelper.average(ctx, encrypted_sum.getData(), extractRawData(partial_plaintexts));
		int[] dims = encrypted_sum.getDims();
		int result_len = Arrays.stream(dims).reduce(1, (x,y) -> x*y);
		DataCharacteristics dc = encrypted_sum.getDataCharacteristics();

		DenseBlock new_dense_block = DenseBlockFactory.createDenseBlock(Arrays.copyOf(raw_result, result_len), dims);
		MatrixBlock new_matrix_block = new MatrixBlock((int)dc.getRows(), (int)dc.getCols(), new_dense_block);
		MatrixObject new_mo = new MatrixObject(Types.ValueType.FP64, OptimizerUtils.getUniqueTempFileName(), new MetaDataFormat(dc, Types.FileFormat.BINARY));
		new_mo.acquireModify(new_matrix_block);
		new_mo.release();
		return new_mo;
	}

	private static byte[][] extractRawData(Encrypted[] data) {
		byte[][] raw_data = new byte[data.length][];
		for (int i = 0; i < data.length; i++) {
			raw_data[i] = data[i].getData();
		}
		return raw_data;
	}

	// TODO: extract an interface for this and use it here
	private static byte[][] extractRawData(PublicKey[] data) {
		byte[][] raw_data = new byte[data.length][];
		for (int i = 0; i < data.length; i++) {
			raw_data[i] = data[i].getData();
		}
		return raw_data;
	}
}
