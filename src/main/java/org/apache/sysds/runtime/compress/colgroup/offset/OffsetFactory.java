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
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public final class OffsetFactory {

	static final Log LOG = LogFactory.getLog(OffsetFactory.class.getName());

	private OffsetFactory() {
		// Empty private constructor.
	}

	/** The specific underlying types of offsets. */
	public enum OFF_TYPE {
		BYTE, CHAR
	}

	/** Specialized types of underlying offsets. */
	public enum OFF_TYPE_SPECIALIZATIONS {
		BYTE, CHAR, SINGLE_OFFSET, TWO_OFFSET, EMPTY
	}

	/**
	 * Main factory pattern creator for Offsets.
	 * 
	 * Note this creator is unsafe it is assumed that the input index list only contain sequential non duplicate
	 * incrementing values.
	 * 
	 * @param indexes List of indexes, that is assumed to be sorted and have no duplicates
	 * @return AOffset object containing offsets to the next value.
	 */
	public static AOffset createOffset(int[] indexes) {
		return createOffset(indexes, 0, indexes.length);
	}

	/**
	 * Create the offsets based on our primitive IntArrayList.
	 * 
	 * Note this creator is unsafe it is assumed that the input index list only contain sequential non duplicate
	 * incrementing values.
	 * 
	 * @param indexes The List of indexes, that is assumed to be sorted and have no duplicates
	 * @return AOffset object containing offsets to the next value.
	 */
	public static AOffset createOffset(IntArrayList indexes) {
		return createOffset(indexes.extractValues(), 0, indexes.size());
	}

	/**
	 * try to create a specific type of offset.
	 * 
	 * @param indexes the List of indexes, that is assumed to be sorted and have no duplicates
	 * @param type    The type requested.
	 * @return The return offset
	 */
	public static AOffset createOffset(int[] indexes, OFF_TYPE type) {
		return createOffset(indexes, 0, indexes.length, type);
	}

	/**
	 * Create a Offset based on a subset of the indexes given.
	 * 
	 * This is useful if the input is created from a CSR matrix, since it allows us to not reallocate the indexes[] but
	 * use the shared indexes from the entire CSR representation.
	 * 
	 * Note this creator is unsafe it is assumed that the input index list only contain sequential non duplicate
	 * incrementing values.
	 * 
	 * @param indexes The indexes from which to take the offsets.
	 * @param apos    The position to start looking from in the indexes.
	 * @param alen    The position to end looking at in the indexes.
	 * @return A new Offset.
	 */
	public static AOffset createOffset(int[] indexes, int apos, int alen) {
		try {
			if(indexes == null)
				throw new DMLCompressionException("Invalid null indexes input");
			final int endLength = alen - apos - 1;
			if(endLength < 0)
				return new OffsetEmpty();
			else if(indexes[0] < 0)
				throw new DMLCompressionException("Invalid negative offset");
			else if(endLength == 0) // means size of 1 since we store the first offset outside the list
				return new OffsetSingle(indexes[apos]);
			else if(endLength == 1)
				return new OffsetTwo(indexes[apos], indexes[apos + 1]);

			final int minValue = indexes[apos];
			final int maxValue = indexes[alen - 1];
			final int range = maxValue - minValue;
			// -1 because one index is skipped using a first idex allocated as a int.

			final int correctionByte = correctionByte(range, endLength);
			final int correctionChar = correctionChar(range, endLength);

			final long byteSize = OffsetByte.estimateInMemorySize(endLength + correctionByte);
			final long charSize = OffsetChar.estimateInMemorySize(endLength + correctionChar);

			if(byteSize < charSize)
				return createByte(indexes, apos, alen);
			else
				return createChar(indexes, apos, alen);
		}
		catch(Exception e) {
			if(indexes == null)
				throw e;
			for(int i = apos + 1; i < alen; i++) {
				if(indexes[i] <= indexes[i - 1]) {
					String message = "Invalid input to create offset, all values should be continuously increasing.\n";
					message += "Index " + (i - 1) + " and Index " + i + " are wrong with values: " + indexes[i - 1] + " and "
						+ indexes[i];
					throw new DMLCompressionException(message, e);
				}
			}
			throw new DMLCompressionException(
				"Failed to create offset with input:" + Arrays.toString(indexes) + " Apos: " + apos + " Alen: " + alen, e);
		}
	}

	public static AOffset createOffset(int[] indexes, int apos, int alen, OFF_TYPE type) {
		final int indexesLength = alen - apos;
		if(indexesLength <= 0)
			return new OffsetEmpty();
		else if(indexesLength == 1)
			return new OffsetSingle(indexes[apos]);
		else if(indexesLength == 2)
			return new OffsetTwo(indexes[apos], indexes[apos + 1]);
		else if(type == OFF_TYPE.BYTE)
			return createByte(indexes, 0, indexes.length);
		else
			return createChar(indexes, 0, indexes.length);
	}

	/**
	 * Read in AOffset from the DataInput.
	 * 
	 * @param in DataInput to read from
	 * @return The AOffset data instance
	 * @throws IOException If the DataInput fails reading in the variables
	 */
	public static AOffset readIn(DataInput in) throws IOException {
		OFF_TYPE_SPECIALIZATIONS t = OFF_TYPE_SPECIALIZATIONS.values()[in.readByte()];
		switch(t) {
			case EMPTY:
				return OffsetEmpty.readFields(in);
			case SINGLE_OFFSET:
				return OffsetSingle.readFields(in);
			case TWO_OFFSET:
				return OffsetTwo.readFields(in);
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
		if(size == 0) // If this is the case, then the compression results in constant col groups
			return OffsetEmpty.estimateInMemorySize();
		else if(size == 1)
			return OffsetSingle.estimateInMemorySize();
		else if(size == 2)
			return OffsetTwo.estimateInMemorySize();

		final int avgDiff = nRows / size;
		if(avgDiff < 256) {
			final int correctionByte = correctionByte(nRows, size);
			return OffsetByte.estimateInMemorySize(size - 1 + correctionByte);
		}
		else {
			final int correctionChar = correctionChar(nRows, size);
			return OffsetChar.estimateInMemorySize(size - 1 + correctionChar);
		}
	}

	public static int correctionByte(int nRows, int size) {
		return Math.max((nRows - size * 256), 0) / 256;
	}

	public static int correctionChar(int nRows, int size) {
		return Math.max((nRows - size * Character.MAX_VALUE), 0) / Character.MAX_VALUE;
	}

	private static AOffset createByte(int[] indexes, int apos, int alen) {
		final int indexesLength = alen - apos;

		int endSize = 0;
		int offsetToFirst = indexes[apos];
		int offsetToLast = indexes[alen - 1];
		int ov = offsetToFirst;
		// find the size of the array
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			endSize += 1 + (nv - ov - 1) / OffsetByte.maxV;
			ov = nv;
		}

		boolean noZero = endSize == indexesLength - 1;
		byte[] offsets = new byte[endSize];
		ov = offsetToFirst;
		int p = 0;

		// populate the array
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			final int offsetSize = nv - ov;
			if(offsetSize <= 0)
				throw new DMLCompressionException("Invalid offset");
			final int div = offsetSize / OffsetByte.maxV;
			final int mod = offsetSize % OffsetByte.maxV;
			if(mod == 0) {
				p += div - 1; // skip values
				offsets[p++] = (byte) OffsetByte.maxV;
			}
			else {
				p += div; // skip values
				offsets[p++] = (byte) (mod);
			}

			ov = nv;
		}

		boolean noOverHalf = getNoOverHalf(offsets);
		return new OffsetByte(offsets, offsetToFirst, offsetToLast, indexesLength, noOverHalf, noZero);
	}

	private static AOffset createChar(int[] indexes, int apos, int alen) {

		int endSize = 0;
		int offsetToFirst = indexes[apos];
		int offsetToLast = indexes[alen - 1];
		int ov = offsetToFirst;
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			endSize += 1 + (nv - ov - 1) / OffsetChar.maxV;
			ov = nv;
		}
		boolean noZero = endSize == alen - apos - 1;
		char[] offsets = new char[endSize];
		ov = offsetToFirst;
		int p = 0;
		for(int i = apos + 1; i < alen; i++) {
			final int nv = indexes[i];
			final int offsetSize = (nv - ov);
			if(offsetSize <= 0)
				throw new DMLCompressionException("Invalid offset");
			final int div = offsetSize / OffsetChar.maxV;
			final int mod = offsetSize % OffsetChar.maxV;
			if(mod == 0) {
				p += div - 1; // skip values
				offsets[p++] = (char) OffsetChar.maxV;
			}
			else {
				p += div; // skip values
				offsets[p++] = (char) (mod);
			}
			ov = nv;
		}
		return new OffsetChar(offsets, offsetToFirst, offsetToLast, noZero);
	}

	protected static boolean getNoOverHalf(byte[] off) {
		boolean noOverHalf = true;
		for(byte b : off)
			if(b < 1) {
				noOverHalf = false;
				break;
			}
		return noOverHalf;
	}

	protected static boolean getNoZero(byte[] off) {
		boolean noZero = true;
		for(byte b : off)
			if(b == 0) {
				noZero = false;
				break;
			}
		return noZero;
	}

	protected static boolean getNoZero(char[] off) {
		boolean noZero = true;
		for(char b : off)
			if(b == 0) {
				noZero = false;
				break;
			}
		return noZero;
	}

}
