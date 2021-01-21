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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.Bitmap;
import org.apache.sysds.runtime.compress.utils.BitmapLossy;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

/**
 * Static functions for encoding bitmaps in various ways.
 */
public class BitmapLossyEncoder {

	private static final Log LOG = LogFactory.getLog(BitmapLossyEncoder.class.getName());

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
	 * @param ubm The Uncompressed version of the bitmap.
	 * @return A bitmap.
	 */
	public static ABitmap makeBitmapLossy(Bitmap ubm) {
		final double[] fp = ubm.getValues();
		if(fp.length == 0) {
			return ubm;
		}
		Stats stats = new Stats(fp);
		// TODO make better decisions than just a 8 Bit encoding.
		if(Double.isInfinite(stats.max) || Double.isInfinite(stats.min)) {
			LOG.warn("Defaulting to incompressable colGroup");
			return ubm;
		}
		else {
			return make8BitLossy(ubm, stats);
		}
	}

	/**
	 * Make the specific 8 bit encoding version of a bitmap.
	 * 
	 * @param ubm   The uncompressed Bitmap.
	 * @param stats The statistics associated with the bitmap.
	 * @return a lossy bitmap.
	 */
	private static BitmapLossy make8BitLossy(Bitmap ubm, Stats stats) {
		final double[] fp = ubm.getValues();
		int numCols = ubm.getNumColumns();
		double scale = get8BitScale(stats.min, stats.max);
		byte[] scaledValues = scaleValuesToByte(fp, scale);
		if(numCols == 1)
			return makeBitmapLossySingleCol(ubm, scaledValues, scale);
		else
			return makeBitmapLossyMultiCol(ubm, scaledValues, scale);

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
	 * @return The Lossy bitmap.
	 */
	private static BitmapLossy makeBitmapLossySingleCol(Bitmap ubm, byte[] scaledValues, double scale) {

		// Using Linked Hashmap to preserve the sorted order.
		Map<Byte, Queue<IntArrayList>> values = new LinkedHashMap<>();
		Map<Byte, Integer> lengths = new HashMap<>();

		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();
		int numZeroGroups = ubm.getZeroCounts();
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
			else {
				numZeroGroups++;
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
			return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, numZeroGroups, scaledValuesReduced, scale);
		}
		else {
			return new BitmapLossy(ubm.getNumColumns(), fullSizeOffsetsLists, numZeroGroups, scaledValues, scale);
		}
	}

	/**
	 * Multi column instance of makeBitmapLossySingleCol
	 * 
	 * @param ubm          The original uncompressed bitmap.
	 * @param scaledValues The scaled values to map into.
	 * @param scale        The scale in use.
	 * @return The Lossy bitmap.
	 */
	private static BitmapLossy makeBitmapLossyMultiCol(Bitmap ubm, byte[] scaledValues, double scale) {
		int numColumns = ubm.getNumColumns();
		Map<List<Byte>, Queue<IntArrayList>> values = new HashMap<>();
		Map<List<Byte>, Integer> lengths = new HashMap<>();
		IntArrayList[] fullSizeOffsetsLists = ubm.getOffsetList();
		int numZeroGroups = ubm.getZeroCounts();
		boolean allZero = true;
		boolean somethingToMerge = false;
		for(int idx = 0; idx < scaledValues.length; idx += numColumns) {
			List<Byte> array = new ArrayList<>();
			for(int off = 0; off < numColumns; off++) {
				allZero = scaledValues[idx + off] == 0 && allZero;
				array.add(scaledValues[idx + off]);
			}

			numZeroGroups += allZero ? 1 : 0;
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

		// HACK; we make sure that the first sparse unsafe operation assume
		// that we have entries with zero values. This makes the first sparse
		// unsafe operation slightly slower, if the input compressed matrix is
		// fully dense, aka containing no zero values.
		// This is required for multi-column colGroups.
		numZeroGroups = numZeroGroups + 1;

		if(somethingToMerge) {

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
				Queue<IntArrayList> q = ent.getValue();
				if(q.size() == 1) {
					newOffsetsLists[idx] = q.remove();
				}
				else {
					newOffsetsLists[idx] = mergeOffsets(q, new int[lengths.get(key)]);
				}
				idx++;
			}

			return new BitmapLossy(ubm.getNumColumns(), newOffsetsLists, numZeroGroups, scaledValuesReduced, scale);
		}
		else {
			return new BitmapLossy(ubm.getNumColumns(), fullSizeOffsetsLists, numZeroGroups, scaledValues, scale);
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
			for(int i = 0; i < h.size(); i++) {
				res[indexStart++] = v[i];
			}
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
		byte[] res = getMemLocalByteArray(fp.length);
		for(int idx = 0; idx < fp.length; idx++) {
			res[idx] = (byte) (Math.round(fp[idx] / scale));
		}
		return res;
	}

	public static ABitmap extractMapFromCompressedSingleColumn(CompressedMatrixBlock m, int columnId, double min,
		double max) {
		// public static void decompressToBlock(MatrixBlock target, int colIndex, List<ColGroup> colGroups) {

		// LOG.error("Lossy Extract Min and Max...");
		// LOG.error(min + " " + max);

		double scale = get8BitScale(min, max);
		final int blkSz = CompressionSettings.BITMAP_BLOCK_SZ;
		Map<Byte, IntArrayList> values = new HashMap<>();
		double[] tmp = getMemLocalDoubleArray(blkSz);
		for(int i = 0; i < m.getNumRows(); i += blkSz) {
			if(i > 0)
				Arrays.fill(tmp, 0);

			ColGroup.decompressColumnToBlock(tmp,
				columnId,
				i ,
				Math.min(m.getNumRows(), (i +  blkSz)),
				m.getColGroups());

			byte[] scaledValues = scaleValuesToByte(tmp, scale);
			for(int j = 0, off = i; j < Math.min(m.getNumRows(), (i + blkSz)) - i; j++, off++) {
				byte key = scaledValues[j];
				if(values.containsKey(key))
					values.get(key).appendValue(off);
				else
					values.put(key, new IntArrayList(off));
			}
		}

		IntArrayList[] newOffsetsLists = new IntArrayList[values.keySet().size()];
		byte[] scaledValuesReduced = new byte[values.keySet().size()];
		Iterator<Entry<Byte, IntArrayList>> x = values.entrySet().iterator();
		int idx = 0;
		while(x.hasNext()) {
			Entry<Byte, IntArrayList> ent = x.next();
			scaledValuesReduced[idx] = ent.getKey().byteValue();
			newOffsetsLists[idx] = ent.getValue();
			idx++;
		}

		return new BitmapLossy(1, newOffsetsLists, 0, scaledValuesReduced, scale);
		// return BitmapLossyEncoder.makeBitmapLossy(BitmapEncoder.extractBitmap(new int[1], tmp, true));
	}

	private static byte[] getMemLocalByteArray(int length){
		byte[] ar =  memPoolByteArray.get();
		if(ar!= null && ar.length >= length)
			return ar;
		else{
			memPoolByteArray.set(new byte[length]);
			return memPoolByteArray.get();
		}
	}

	private static double[] getMemLocalDoubleArray(int length){
		double[] ar =  memPoolDoubleArray.get();
		if(ar!= null && ar.length >= length)
			return ar;
		else{
			memPoolDoubleArray.set(new double[length]);
			return memPoolDoubleArray.get();
		}
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
