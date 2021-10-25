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

package org.apache.sysds.runtime.compress.colgroup.offset;

import java.io.DataInput;
import java.io.IOException;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public interface OffsetFactory {

	// static final Log LOG = LogFactory.getLog(OffsetFactory.class.getName());

	public enum OFF_TYPE {
		BYTE, CHAR
	}

	/**
	 * Main factory pattern creator for Offsets.
	 * 
	 * @param indexes List of indexes, that is assumed to be sorted and have no duplicates
	 * @return AOffset object containing offsets to the next value.
	 */
	public static AOffset create(int[] indexes) {
		return create(indexes, 0, indexes.length);
	}

	/**
	 * Create a Offset based on a subset of the indexes given.
	 * 
	 * This is useful if the input is created from a CSR matrix, since it allows us to not reallocate the indexes[] but
	 * use the shared indexes from the entire CSR representation.
	 * 
	 * @param indexes The indexes from which to take the offsets.
	 * @param apos    The position to start looking from in the indexes.
	 * @param alen    The position to end looking at in the indexes.
	 * @return A new Offset.
	 */
	public static AOffset create(int[] indexes, int apos, int alen) {
		final int maxValue = indexes[alen - 1];
		if(maxValue < 0)
			throw new DMLCompressionException("Invalid sizes given");
		final int endLength = alen - apos;
		final float avgDist = (float) maxValue / endLength;
		if(avgDist < 256)
			return new OffsetByte(indexes, apos, alen);
		else
			return new OffsetChar(indexes, apos, alen);
	}

	/**
	 * Read in AOffset from the DataInput.
	 * 
	 * @param in DataInput to read from
	 * @return The AOffset data instance
	 * @throws IOException If the DataInput fails reading in the variables
	 */
	public static AOffset readIn(DataInput in) throws IOException {
		OFF_TYPE t = OFF_TYPE.values()[in.readByte()];
		switch(t) {
			case BYTE:
				return OffsetByte.readFields(in);
			case CHAR:
			default:
				return OffsetChar.readFields(in);
		}
	}

	/**
	 * Avg diff only works assuming a normal distribution of the offsets. This means that if we have 1000 rows and 100
	 * offsets, it is assumed that on average the distance between elements is 10.
	 * 
	 * Optionally todo is to add some number of size if the average distance is almost the same as the max value of the
	 * OffsetLists. this would add to the estimated size and approximate better the real compression size. It would also
	 * then handle edge cases better.
	 * 
	 * @param size  The estimated number of offsets
	 * @param nRows The number of rows.
	 * @return The estimated size of an offset given the number of offsets and rows.
	 */
	public static long estimateInMemorySize(int size, int nRows) {
		if(size < 0 || nRows < 0)
			throw new DMLCompressionException("Invalid sizes given: " + size + "  " + nRows);
		else if(size == 0)
			return 8; // If this is the case, then the compression results in constant col groups
		else {
			final int avgDiff = nRows / size;
			if(avgDiff < 256)
				return OffsetByte.getInMemorySize(size - 1);
			else
				return OffsetChar.getInMemorySize(size - 1);
		}
	}
}
