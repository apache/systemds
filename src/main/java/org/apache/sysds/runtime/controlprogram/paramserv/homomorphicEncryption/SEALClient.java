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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.NativeHEHelper;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.instructions.cp.CiphertextMatrix;
import org.apache.sysds.runtime.instructions.cp.PlaintextMatrix;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.stream.IntStream;

public class SEALClient {
	public SEALClient(byte[] a) {
		// TODO take params here, like slot_count etc.
		ctx = NativeHEHelper.initClient(a);
	}

	// this is a pointer to the context used by all native methods of this class
	private final long ctx;


	/**
	 * generates a partial public key
	 * stores a partial private key corresponding to the partial public key in ctx
	 *
	 * @return the partial public key
	 */
	public PublicKey generatePartialPublicKey() {
		return new PublicKey(NativeHEHelper.generatePartialPublicKey(ctx));
	}

	/**
	 * sets the public key and stores it in ctx
	 *
	 * @param public_key the public key to set
	 */
	public void setPublicKey(PublicKey public_key) {
		NativeHEHelper.setPublicKey(ctx, public_key.getData());
	}

	/**
	 * encrypts one block of data with public key stored statically and returns it
	 * setPublicKey() must have been called before calling this
	 * @param plaintext the MatrixObject to encrypt
	 * @return the encrypted matrix
	 */
	public CiphertextMatrix encrypt(MatrixObject plaintext) {
		MatrixBlock mb = plaintext.acquireReadAndRelease();
		if (mb.isInSparseFormat()) {
			mb.allocateSparseRowsBlock();
			mb.sparseToDense();
		}
		DenseBlock db = mb.getDenseBlock();
		int[] dims = IntStream.range(0, db.numDims()).map(db::getDim).toArray();
		double[] raw_data = mb.getDenseBlockValues();
		return new CiphertextMatrix(dims, plaintext.getDataCharacteristics(), NativeHEHelper.encrypt(ctx, raw_data));
	}

	/**
	 * partially decrypts ciphertext with the partial private key. generatePartialPublicKey() must
	 * have been called before calling this function
	 *
	 * @param ciphertext the ciphertext to partially decrypt
	 * @return the partial decryption of ciphertext
	 */
	public PlaintextMatrix partiallyDecrypt(CiphertextMatrix ciphertext) {
		return new PlaintextMatrix(ciphertext.getDims(), ciphertext.getDataCharacteristics(), NativeHEHelper.partiallyDecrypt(ctx, ciphertext.getData()));
	}
}
