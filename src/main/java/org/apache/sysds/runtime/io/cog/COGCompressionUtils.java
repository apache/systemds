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

package org.apache.sysds.runtime.io.cog;

import org.apache.sysds.runtime.DMLRuntimeException;

import java.io.ByteArrayOutputStream;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

public class COGCompressionUtils {
	/**
	 * Decompresses a byte array that was compressed using the Deflate algorithm
	 * @param compressedData ???
	 * @return ???
	 * @throws DMLRuntimeException ???
	 */
	public static byte[] decompressDeflate(byte[] compressedData) throws DMLRuntimeException {
		// Use the native Java implementation of deflate to decompress the data
		Inflater inflater = new Inflater();
		inflater.setInput(compressedData);

		ByteArrayOutputStream outputStream = new ByteArrayOutputStream(compressedData.length);
		byte[] buffer = new byte[1024];

		while (!inflater.finished()) {
			int decompressedSize = 0;
			try {
				decompressedSize = inflater.inflate(buffer);
			} catch (DataFormatException e) {
				throw new DMLRuntimeException("Failed to decompress tile data", e);
			}
			outputStream.write(buffer, 0, decompressedSize);
		}

		inflater.end();

		return outputStream.toByteArray();
	}
}
