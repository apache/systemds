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

package org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheDataInput;
import org.apache.sysml.runtime.controlprogram.caching.CacheDataOutput;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class PSRpcObject {

	public static final int PUSH = 1;
	public static final int PULL = 2;
	public static final String EMPTY_DATA = "";

	public abstract void deserialize(ByteBuffer buffer) throws IOException;

	public abstract ByteBuffer serialize() throws IOException;


	/**
	 * Deep serialize a list object (currently only support list containing matrices)
	 * @param lo a list object containing only matrices
	 * @return byte array
	 */
	protected byte[] deepSerializeListObject(ListObject lo) throws IOException {
		validateListObject(lo);
		byte[] result = new byte[(int) getListObjectSerializationSize(lo)];
		CacheDataOutput cdo = new CacheDataOutput(result);
		for (int i = 0; i < lo.getLength(); i++) {
			if (lo.isNamedList()) {
				String name = lo.getName(i);
				cdo.writeInt(name.length()); // write name size
				cdo.write(name.getBytes()); // write name
			} else {
				cdo.writeInt(-1); // non-named list
			}
			MatrixObject mo = (MatrixObject) lo.getData().get(i);
			MatrixBlock mb = mo.acquireRead();
			cdo.writeInt((int) mb.getExactSizeOnDisk()); // write matrix size
			mb.write(cdo); // write matrix data
			mo.release();
		}
		return result;
	}

	/**
	 * Get serialization size of a list object
	 * (scheme: size|name|size|matrix)
	 * @param lo list object
	 * @return serialization size
	 */
	private long getListObjectSerializationSize(ListObject lo) {
		long result = 0;
		result += lo.getLength() * Integer.BYTES; // bytes for the size of names
		if (lo.isNamedList()) {
			result += lo.getNames().stream().reduce(0, (size, s) -> size + s.getBytes().length, (i1, i2) -> i1 + i2); // get the name as key
		}
		result += lo.getLength() * Integer.BYTES; // bytes for the byte size of matrix
		result += lo.getData().stream()
			.reduce(0L, (size, data) -> size + ((MatrixObject) data).acquireRead().getExactSizeOnDisk(), (l1, l2) -> l1 + l2);
		return result;
	}

	private void validateListObject(ListObject lo) {
		for (Data d : lo.getData()) {
			if (!(d instanceof MatrixObject)) {
				throw new DMLRuntimeException(String.format("Paramserv func: not support deep serializing data %s which is not matrix.",
					d.getDebugName()));
			}
		}
	}

	protected ListObject parseListObject(ByteBuffer buffer) throws IOException {
		if (!buffer.hasRemaining()) {
			return null;
		}
		List<Data> data = new LinkedList<>();
		List<String> names = null;
		while (buffer.hasRemaining()) {
			// parse the name
			int nameSize = buffer.getInt();
			if (nameSize != -1) {
				byte[] nameBytes = new byte[nameSize];
				buffer.get(nameBytes);
				String name = new String(nameBytes, "ISO-8859-1");
				if (names == null) {
					names = new LinkedList<>();
				}
				names.add(name);
			}

			// parse the matrix
			int matrixSize = buffer.getInt();
			byte[] dataBytes = new byte[matrixSize];
			buffer.get(dataBytes);
			CacheDataInput cdi = new CacheDataInput(dataBytes);
			MatrixBlock mb = new MatrixBlock();
			mb.readFields(cdi);
			MatrixObject mo = ParamservUtils.newMatrixObject(mb);
			mo.enableCleanup(false);
			data.add(mo);
		}
		return new ListObject(data, names);
	}
}
