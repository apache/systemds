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


package org.apache.sysds.runtime.io.hdf5.checksum;

import org.apache.sysds.runtime.io.hdf5.exceptions.HdfChecksumMismatchException;

import java.nio.ByteBuffer;

public final class ChecksumUtils {

	private ChecksumUtils() {
		throw new AssertionError("No instances of ChecksumUtils");
	}

	public static void validateChecksum(ByteBuffer buffer) {
		byte[] bytes = new byte[buffer.limit() - 4];
		buffer.get(bytes);
		int calculatedChecksum = checksum(bytes);
		int storedChecksum = buffer.getInt();
		if(calculatedChecksum != storedChecksum) {
			throw new HdfChecksumMismatchException(storedChecksum, calculatedChecksum);
		}

	}

	public static int checksum(ByteBuffer buffer) {
		return JenkinsLookup3HashLittle.hash(buffer);
	}

	public static int checksum(byte[] bytes) {
		return JenkinsLookup3HashLittle.hash(bytes);
	}

}
