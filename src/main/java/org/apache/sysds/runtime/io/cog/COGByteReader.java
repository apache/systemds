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

import java.io.BufferedInputStream;
import java.io.IOException;

/**
 * This class is used by the COGReader to read bytes from a BufferedInputStream.
 * It is wrapper that keeps track of the bytes read and can therefore be used to
 * easily go to specific offsets.
 */
public class COGByteReader {
	private long totalBytesRead;
	private BufferedInputStream bis;
	private long readlimit = 0;

	public COGByteReader(BufferedInputStream bis) {
		this.bis = bis;
		totalBytesRead = 0;
	}

	public COGByteReader(BufferedInputStream bis, int totalBytesRead) {
		this.bis = bis;
		this.totalBytesRead = totalBytesRead;
	}

	public long getTotalBytesRead() {
		return totalBytesRead;
	}

	public void setTotalBytesRead(int totalBytesRead) {
		this.totalBytesRead = totalBytesRead;
	}

	/**
	 * Reads a given number of bytes from the BufferedInputStream.
	 * Increments the totalBytesRead counter by the number of bytes read.
	 * @param length ???
	 * @return ???
	 */
	public byte[] readBytes(int length) {
		byte[] header = new byte[length];
		try {
			bis.read(header);
			totalBytesRead += length;
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
		return header;
	}

	/**
	 * Reads a given number of bytes from the BufferedInputStream.
	 * Increments the totalBytesRead counter by the number of bytes read.
	 * @param length ???
	 * @return  ???
	 */
	public byte[] readBytes(long length) {
		// TODO: When properly implementing BigTIFF, this could be a problem when not being able to skip bytes
		// In BigTIFF the offset can be larger than maxInt which isn't a problem for skipping bytes
		// but could be a problem when the tiles are not sequential in the file and we need to jump back
		// to a previous position (where we can't use skip).
		if (length > Integer.MAX_VALUE) {
			throw new DMLRuntimeException("Cannot read more than Integer.MAX_VALUE bytes at once");
		}
		return readBytes((int) length);
	}

	/**
	 * Offers the same functionality as BufferedInputStream.mark.
	 * Allows for returning to a previous point if the readlimit is not exceeded.
	 * @param readlimit ???
	 */
	public void mark(long readlimit) {
		this.readlimit = readlimit;
		bis.mark((int) readlimit + 1);
	}

	/**
	 * Offers the same functionality as BufferedInputStream.reset.
	 * Resets the stream to the last marked position.
	 * @throws DMLRuntimeException ???
	 */
	public void reset() throws DMLRuntimeException {
		try {
			bis.reset();
			totalBytesRead -= this.readlimit;
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	/**
	 * Skips a given number of bytes without reading them.
	 * Useful for jumping to specific offsets
	 * @param n Number of bytes to skip
	 * @throws DMLRuntimeException ???
	 */
	public void skipBytes(long n) throws DMLRuntimeException {
		try {
			long skipped = bis.skip(n);
			if (skipped != n) {
				throw new DMLRuntimeException("Could not skip " + n + " bytes, only skipped " + skipped + " bytes");
			}
			totalBytesRead += n;
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
}
