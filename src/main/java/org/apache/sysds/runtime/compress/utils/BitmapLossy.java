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

package org.apache.sysds.runtime.compress.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;

import org.apache.commons.lang.NotImplementedException;

/**
 * Uncompressed but Quantized representation of contained data.
 */
public final class BitmapLossy extends AbstractBitmap {

	/**
	 * Distinct values that appear in the column. Linearized as value groups <v11 v12> <v21 v22>.
	 */
	private final byte[] _values;
	private final double _scale;

	public BitmapLossy(int numCols, IntArrayList[] offsetsLists, int numZeroGroups, byte[] values, double scale) {
		super(numCols, offsetsLists, numZeroGroups);
		_values = values;
		_scale = scale;
	}

	public static AbstractBitmap makeBitmapLossy(Bitmap ubm) {
		int numCols = ubm.getNumColumns();
		double[] fp = ubm.getValues();
		double scale = getScale(fp);
		if(Double.isNaN(scale)) {
			LOG.warn("Defaulting to incompressable colGroup");
			return ubm;
		}
		else {
			byte[] scaledValues = scaleValues(fp, scale);
			if(numCols == 1) {
				return makeBitmapLossySingleCol(ubm, scaledValues, scale);
			}
			else {
				return makeBitmapLossyMultiCol(ubm, scaledValues, scale);
			}
		}

	}

	private static AbstractBitmap makeBitmapLossySingleCol(Bitmap ubm, byte[] scaledValues, double scale) {

		Map<Byte, Queue<IntArrayList>> values = new HashMap<>();
		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();
		int numZeroGroups = ubm.getZeroCounts();
		for(int idx = 0; idx < scaledValues.length; idx++) {
			if(scaledValues[idx] != 0) { // Throw away zero values.
				if(values.containsKey(scaledValues[idx])) {
					values.get(scaledValues[idx]).add(fullSizeOffsetsLists[idx]);
				}
				else {
					Queue<IntArrayList> offsets = new LinkedList<IntArrayList>();
					offsets.add(fullSizeOffsetsLists[idx]);
					values.put(scaledValues[idx], offsets);
				}
			}
			else {
				numZeroGroups++;
			}
		}
		byte[] scaledValuesReduced = new byte[values.keySet().size()];
		IntArrayList[] newOffsetsLists = new IntArrayList[values.keySet().size()];
		Iterator<Entry<Byte, Queue<IntArrayList>>> x = values.entrySet().iterator();
		int idx = 0;
		while(x.hasNext()) {
			Entry<Byte, Queue<IntArrayList>> ent = x.next();
			scaledValuesReduced[idx] = ent.getKey().byteValue();
			newOffsetsLists[idx] = mergeOffsets(ent.getValue());
			idx++;
		}
		return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, numZeroGroups, scaledValuesReduced, scale);
	}

	private static AbstractBitmap makeBitmapLossyMultiCol(Bitmap ubm, byte[] scaledValues, double scale) {
		int numColumns = ubm.getNumColumns();
		Map<List<Byte>, Queue<IntArrayList>> values = new HashMap<>();
		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();
		int numZeroGroups = ubm.getZeroCounts();
		boolean allZero = true;
		for(int idx = 0; idx < scaledValues.length; idx += numColumns) {
			List<Byte> array = new ArrayList<>();
			for(int off = 0; off < numColumns; off++) {
				allZero = scaledValues[idx + off] == 0 && allZero;
				array.add(scaledValues[idx + off]);
			}
			
			numZeroGroups += allZero ? 1 : 0;
			if(!allZero) {
				if(values.containsKey(array)) {
					values.get(array).add(fullSizeOffsetsLists[idx / numColumns]);
				}
				else {
					Queue<IntArrayList> offsets = new LinkedList<IntArrayList>();
					offsets.add(fullSizeOffsetsLists[idx / numColumns]);
					values.put(array, offsets);
				}
				// LOG.error(array);
			}
			allZero = true;
		}
		// LOG.error(array);
		// LOG.error(values);


		byte[] scaledValuesReduced = new byte[values.keySet().size() * numColumns];
		IntArrayList[] newOffsetsLists = new IntArrayList[values.keySet().size()];
		Iterator<Entry<List<Byte>, Queue<IntArrayList>>> x = values.entrySet().iterator();
		int idx = 0;
		while(x.hasNext()) {
			Entry<List<Byte>, Queue<IntArrayList>> ent = x.next();
			List<Byte> key = ent.getKey();
			int row = idx * numColumns;
			for(int off = 0; off < numColumns; off++) {
				scaledValuesReduced[row + off] = key.get(off);
			}
			newOffsetsLists[idx] = mergeOffsets(ent.getValue());
			idx++;
		}
		// LOG.error(Arrays.toString(scaledValuesReduced));
		// try {
		// 	Thread.sleep(1000);
		// }
		// catch(InterruptedException e) {
		// 	// TODO Auto-generated catch block
		// 	e.printStackTrace();
		// }
		return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, numZeroGroups, scaledValuesReduced, scale);
	}

	/**
	 * Get the scale for the given double array.
	 * 
	 * @param fp A array of double values
	 * @return a scale to scale to range [-127, 127]
	 */
	public static double getScale(double[] fp) {
		DoubleSummaryStatistics stat = Arrays.stream(fp).summaryStatistics();
		double max = Math.abs(Math.max(stat.getMax(), Math.abs(stat.getMin())));
		double scale;
		if(Double.isInfinite(max)) {
			LOG.warn("Invalid Column, can't quantize Infinite value.");
			return Double.NaN;
		}
		else if(max == 0) { // The column group is filled with 0.
			scale = 1;
		}
		else {
			scale = max / (double) (Byte.MAX_VALUE);
		}
		return scale;
	}

	/**
	 * Get all values without unnecessary allocations and copies.
	 * 
	 * @return dictionary of value tuples
	 */
	public byte[] getValues() {
		return _values;
	}

	/**
	 * Obtain tuple of column values associated with index.
	 * 
	 * @param ix index of a particular distinct value
	 * @return the tuple of column values associated with the specified index
	 */
	public byte[] getValues(int ix) {
		return Arrays.copyOfRange(_values, ix * _numCols, (ix + 1) * _numCols);
	}

	public double getScale() {
		return _scale;
	}

	/**
	 * Obtain number of distinct values in the column.
	 * 
	 * @return number of distinct values in the column; this number is also the number of bitmaps, since there is one
	 *         bitmap per value
	 */
	public int getNumValues() {
		return _values.length / _numCols;
	}

	public IntArrayList getOffsetsList(int ix) {
		return _offsetsLists[ix];
	}

	public long getNumOffsets() {
		long ret = 0;
		for(IntArrayList offlist : _offsetsLists)
			ret += offlist.size();
		return ret;
	}

	public int getNumOffsets(int ix) {
		return _offsetsLists[ix].size();
	}

	@Override
	public void sortValuesByFrequency() {
		// TODO Auto-generated method stub
		throw new NotImplementedException("Not Implemented Sorting of Lossy Bit Map");
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\nValues: " + Arrays.toString(_values));
		sb.append("\ncolumns:" + _numCols);
		sb.append("\nScale:  " + _scale);
		sb.append("\nOffsets:" + Arrays.toString(_offsetsLists));
		return sb.toString();
	}

	// UTIL FUNCTIONS

	private static IntArrayList mergeOffsets(Queue<IntArrayList> offsets) {
		if(offsets.size() == 1) {
			return offsets.remove();
		}
		else {
			IntArrayList h = offsets.remove();
			IntArrayList t = offsets.remove();
			IntArrayList n = mergeOffsets(h, t);
			offsets.add(n);
			return mergeOffsets(offsets);
		}
	}

	private static IntArrayList mergeOffsets(IntArrayList h, IntArrayList t) {
		int lhsSize = h.size(); // Size left
		int rhsSize = t.size(); // Size right
		int[] res = new int[lhsSize + rhsSize]; // Result array.
		int[] lhs = h.extractValues(); // Left hand side values
		int[] rhs = t.extractValues(); // Right hand side values
		int lhsP = 0; // Left hand side pointer
		int rhsP = 0; // Right hand side pointer
		int p = 0; // Pointer in array.
		while(lhsP < lhsSize || rhsP < rhsSize) {
			if(lhsP < lhsSize && (rhsP == rhsSize || lhs[lhsP] < rhs[rhsP])) {
				res[p++] = lhs[lhsP++];
			}
			else {
				res[p++] = rhs[rhsP++];
			}
		}
		return new IntArrayList(res);
	}

	@Override
	public BitmapType getType() {
		return BitmapType.Lossy;
	}

	/**
	 * Utility method to scale all the values in the array to byte range
	 * 
	 * TODO make scaling parallel since each scaling is independent.
	 * 
	 * @param fp    doulbe array to scale
	 * @param scale the scale to apply
	 * @return the scaled values in byte
	 */
	public static byte[] scaleValues(double[] fp, double scale) {
		byte[] res = new byte[fp.length];
		for(int idx = 0; idx < fp.length; idx++) {
			res[idx] = (byte) (fp[idx] / scale);
		}
		return res;
	}
}
