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

package org.apache.sysds.runtime.compress.lib;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * Static functions for encoding bitmaps in various ways.
 */
public class BitmapLossyEncoder {

	// private static final Log LOG = LogFactory.getLog(BitmapLossyEncoder.class.getName());

	private static ThreadLocal<byte[]> memPoolByteArray = new ThreadLocal<byte[]>() {
		@Override
		protected byte[] initialValue() {
			return null;
		}
	};

	private static ThreadLocal<double[]> memPoolDoubleArray = new ThreadLocal<double[]>() {
		@Override
		protected double[] initialValue() {
			return null;
		}
	};

	/**
	 * Given a Bitmap try to make a lossy version of the same bitmap.
	 * 
	 * @param ubm     The Uncompressed version of the bitmap.
	 * @param numRows The number of rows contained in the ubm.
	 * @return A bitmap.
	 */
	public static ABitmap makeBitmapLossy(ABitmap ubm, int numRows) {
		throw new NotImplementedException();
		// final double[] fp = ubm.getValues();
		// if(fp.length == 0) {
		// 	return ubm;
		// }
		// Stats stats = new Stats(fp);
		// // TODO make better decisions than just a 8 Bit encoding.
		// if(Double.isInfinite(stats.max) || Double.isInfinite(stats.min)) {
		// 	LOG.warn("Defaulting to incompressable colGroup");
		// 	return ubm;
		// }
		// else {
		// 	return make8BitLossy(ubm, stats, numRows);
		// }
	}

	/**
	 * Make the specific 8 bit encoding version of a bitmap.
	 * 
	 * @param ubm     The uncompressed Bitmap.
	 * @param stats   The statistics associated with the bitmap.
	 * @param numRows The number of Rows.
	 * @return a lossy bitmap.
	 */
	@SuppressWarnings("unused")
	private static BitmapLossy make8BitLossy(Bitmap ubm, Stats stats, int numRows) {
		final double[] fp = ubm.getValues();
		int numCols = ubm.getNumColumns();
		double scale = get8BitScale(stats.min, stats.max);
		byte[] scaledValues = scaleValuesToByte(fp, scale);
		if(numCols == 1)
			return makeBitmapLossySingleCol(ubm, scaledValues, scale, numRows);
		else
			return makeBitmapLossyMultiCol(ubm, scaledValues, scale, numRows);

	}

	private static double get8BitScale(double min, double max) {
		return Math.max(Math.abs(min), Math.abs(max)) / (double) Byte.MAX_VALUE;
	}

	/**
	 * Make Single column lossy bitmap.
	 * 
	 * This method merges the previous offset lists together to reduce the size.
	 * 
	 * @param ubm          The original uncompressed bitmap.
	 * @param scaledValues The scaled values to map into.
	 * @param scale        The scale in use.
	 * @param numRows      The number of rows in the input.
	 * @return The Lossy bitmap.
	 */
	private static BitmapLossy makeBitmapLossySingleCol(Bitmap ubm, byte[] scaledValues, double scale, int numRows) {

		// Using Linked Hashmap to preserve the sorted order.
		Map<Byte, Queue<IntArrayList>> values = new LinkedHashMap<>();
		Map<Byte, Integer> lengths = new HashMap<>();

		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();
		boolean somethingToMerge = false;

		for(int idx = 0; idx < scaledValues.length; idx++) {
			if(scaledValues[idx] != 0) { // Throw away zero values.
				if(values.containsKey(scaledValues[idx])) {
					values.get(scaledValues[idx]).add(fullSizeOffsetsLists[idx]);
					lengths.put(scaledValues[idx], lengths.get(scaledValues[idx]) + fullSizeOffsetsLists[idx].size());
					somethingToMerge = true;
				}
				else {
					Queue<IntArrayList> offsets = new LinkedList<>();
					offsets.add(fullSizeOffsetsLists[idx]);
					values.put(scaledValues[idx], offsets);
					lengths.put(scaledValues[idx], fullSizeOffsetsLists[idx].size());
				}
			}

		}

		if(somethingToMerge) {
			byte[] scaledValuesReduced = new byte[values.keySet().size()];
			IntArrayList[] newOffsetsLists = new IntArrayList[values.keySet().size()];
			Iterator<Entry<Byte, Queue<IntArrayList>>> x = values.entrySet().iterator();
			int idx = 0;
			while(x.hasNext()) {
				Entry<Byte, Queue<IntArrayList>> ent = x.next();
				scaledValuesReduced[idx] = ent.getKey().byteValue();
				Queue<IntArrayList> q = ent.getValue();
				if(q.size() == 1) {
					newOffsetsLists[idx] = q.remove();
				}
				else {
					newOffsetsLists[idx] = mergeOffsets(q, new int[lengths.get(ent.getKey())]);
				}
				idx++;
			}
			return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, scaledValuesReduced, scale, numRows);
		}
		else
			return new BitmapLossy(ubm.getNumColumns(), fullSizeOffsetsLists, scaledValues, scale, numRows);
	}

	/**
	 * Multi column instance of makeBitmapLossySingleCol
	 * 
	 * @param ubm          The original uncompressed bitmap.
	 * @param scaledValues The scaled values to map into.
	 * @param scale        The scale in use.
	 * @param numRows      The number of rows in each column group
	 * @return The Lossy bitmap.
	 */
	private static BitmapLossy makeBitmapLossyMultiCol(Bitmap ubm, byte[] scaledValues, double scale, int numRows) {
		int numColumns = ubm.getNumColumns();
		Map<List<Byte>, Queue<IntArrayList>> values = new HashMap<>();
		Map<List<Byte>, Integer> lengths = new HashMap<>();
		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();

		boolean allZero = true;
		boolean somethingToMerge = false;
		for(int idx = 0; idx < scaledValues.length; idx += numColumns) {
			List<Byte> array = new ArrayList<>();
			for(int off = 0; off < numColumns; off++) {
				allZero = scaledValues[idx + off] == 0 && allZero;
				array.add(scaledValues[idx + off]);
			}

			if(!allZero) {
				IntArrayList entry = fullSizeOffsetsLists[idx / numColumns];
				if(values.containsKey(array)) {
					values.get(array).add(entry);
					lengths.put(array, lengths.get(array) + entry.size());
					somethingToMerge = true;
				}
				else {
					Queue<IntArrayList> offsets = new LinkedList<>();
					offsets.add(entry);
					values.put(array, offsets);
					lengths.put(array, entry.size());
				}
			}
			allZero = true;
		}

		if(somethingToMerge) {

			byte[] scaledValuesReduced = new byte[values.keySet().size() * numColumns];
			IntArrayList[] newOffsetsLists = new IntArrayList[values.keySet().size()];
			Iterator<Entry<List<Byte>, Queue<IntArrayList>>> x = values.entrySet().iterator();
			int idx = 0;
			while(x.hasNext()) {
				Entry<List<Byte>, Queue<IntArrayList>> ent = x.next();
				List<Byte> key = ent.getKey();
				int row = idx * numColumns;
				for(int off = 0; off < numColumns; off++)
					scaledValuesReduced[row + off] = key.get(off);

				Queue<IntArrayList> q = ent.getValue();
				if(q.size() == 1)
					newOffsetsLists[idx] = q.remove();
				else
					newOffsetsLists[idx] = mergeOffsets(q, new int[lengths.get(key)]);

				idx++;
			}

			return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, scaledValuesReduced, scale, numRows);
		}
		else {
			return new BitmapLossy(ubm.getNumColumns(), fullSizeOffsetsLists, scaledValues, scale, numRows);
		}
	}

	/**
	 * Merge method to join together offset lists.
	 * 
	 * @param offsets The offsets to join
	 * @param res     The result int array to put the values into. This has to be allocated to the joined size of all
	 *                the input offsetLists
	 * @return The merged offsetList.
	 */
	private static IntArrayList mergeOffsets(Queue<IntArrayList> offsets, int[] res) {
		int indexStart = 0;
		while(!offsets.isEmpty()) {
			IntArrayList h = offsets.remove();
			int[] v = h.extractValues();
			for(int i = 0; i < h.size(); i++)
				res[indexStart++] = v[i];
		}
		Arrays.sort(res);
		return new IntArrayList(res);
	}

	/**
	 * Utility method to scale all the values in the array to byte range
	 * 
	 * @param fp    double array to scale
	 * @param scale the scale to apply
	 * @return the scaled values in byte
	 */
	private static byte[] scaleValuesToByte(double[] fp, double scale) {
		byte[] res = getMemLocalByteArray(fp.length, false);
		for(int idx = 0; idx < fp.length; idx++)
			res[idx] = (byte) (Math.round(fp[idx] / scale));

		return res;
	}

	private static byte[] getMemLocalByteArray(int length, boolean clean) {
		byte[] ar = memPoolByteArray.get();
		if(ar != null && ar.length >= length) {
			if(clean)
				for(int i = 0; i < length; i++)
					ar[i] = 0;
			return ar;
		}
		else {
			memPoolByteArray.set(new byte[length]);
			return memPoolByteArray.get();
		}
	}

	@SuppressWarnings("unused")
	private static double[] getMemLocalDoubleArray(int length, boolean clean) {
		double[] ar = memPoolDoubleArray.get();
		if(ar != null && ar.length >= length) {
			if(clean)
				Arrays.fill(ar, 0.0);
			return ar;
		}
		else {
			memPoolDoubleArray.set(new double[length]);
			return memPoolDoubleArray.get();
		}
	}

	public static void cleanMemPools() {
		memPoolByteArray.remove();
		memPoolDoubleArray.remove();
	}

	/**
	 * Statistics class to analyse what compression plan to use.
	 */
	private static class Stats {
		protected double max;
		protected double min;
		protected double minDelta;
		protected double maxDelta;
		protected boolean sameDelta;

		@SuppressWarnings("unused")
		public Stats(double[] fp) {
			max = Double.NEGATIVE_INFINITY;
			min = Double.POSITIVE_INFINITY;
			maxDelta = Double.NEGATIVE_INFINITY;
			minDelta = Double.POSITIVE_INFINITY;
			sameDelta = true;
			if(fp.length > 1) {

				double delta = fp[0] - fp[1];
				for(int i = 0; i < fp.length - 1; i++) {
					if(fp[i] > max)
						max = fp[i];
					if(fp[i] < min)
						min = fp[i];
					double ndelta = fp[i] - fp[i + 1];
					if(delta < minDelta) {
						minDelta = delta;
					}
					if(delta > maxDelta) {
						maxDelta = delta;
					}
					if(sameDelta && Math.abs(delta - ndelta) <= delta * 0.00000001) {
						sameDelta = false;
					}
					delta = ndelta;
				}
				if(fp[fp.length - 1] > max)
					max = fp[fp.length - 1];
				if(fp[fp.length - 1] < min)
					min = fp[fp.length - 1];
			}
			else {
				max = fp[0];
				min = fp[0];
				maxDelta = 0;
				minDelta = 0;
			}
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("Stats{" + this.hashCode() + "}");
			sb.append(" max: " + max);
			sb.append(" min: " + min);
			sb.append(" minΔ: " + minDelta);
			sb.append(" maxΔ: " + maxDelta);
			sb.append(" sameΔ: " + maxDelta);
			return sb.toString();
		}
	}
}
