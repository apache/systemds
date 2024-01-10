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

package org.apache.sysds.runtime.util;

import java.util.Arrays;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Utilities for sorting, primarily used for SparseRows.
 * 
 */
public class SortUtils {
	public static boolean isSorted(int start, int end, int[] indexes) {
		boolean ret = true;
		for(int i = start + 1; i < end && ret; i++)
			ret &= (indexes[i - 1] < indexes[i]);
		return ret;
	}

	public static boolean isSorted(int start, int end, double[] values) {
		boolean ret = true;
		for(int i = start + 1; i < end && ret; i++)
			ret &= (values[i - 1] < values[i]);
		return ret;
	}

	public static boolean isSorted(MatrixBlock in) {
		return in.isInSparseFormat() ? false : !in.isAllocated() ? true : isSorted(0,
			in.getNumRows() * in.getNumColumns(), in.getDenseBlockValues());
	}

	public static int compare(double[] d1, double[] d2) {
		if(d1 == null || d2 == null)
			throw new RuntimeException("Invalid invocation w/ null parameter.");
		int ret = Long.compare(d1.length, d2.length);
		if(ret != 0)
			return ret;
		for(int i = 0; i < d1.length; i++) {
			ret = Double.compare(d1[i], d2[i]);
			if(ret != 0)
				return ret;
		}
		return 0;
	}

	/**
	 * In-place sort of two arrays, only indexes is used for comparison and values of same position are sorted
	 * accordingly.
	 * 
	 * @param start   starting index
	 * @param end     ending index
	 * @param indexes array of indexes to sort by
	 * @param values  double array of values to sort
	 */
	public static void sortByIndex(int start, int end, int[] indexes, double[] values) {
		int tempIx;
		double tempVal;

		int length = end - start;
		if(length < 7) {
			for(int i = start + 1; i < end; i++) {
				for(int j = i; j > start && indexes[j - 1] > indexes[j]; j--) {
					tempIx = indexes[j];
					indexes[j] = indexes[j - 1];
					indexes[j - 1] = tempIx;
					tempVal = values[j];
					values[j] = values[j - 1];
					values[j - 1] = tempVal;
				}
			}
			return;
		}
		int middle = (start + end) / 2;
		if(length > 7) {
			int bottom = start;
			int top = end - 1;
			if(length > 40) {
				length /= 8;
				bottom = med3(indexes, bottom, bottom + length, bottom + (2 * length));
				middle = med3(indexes, middle - length, middle, middle + length);
				top = med3(indexes, top - (2 * length), top - length, top);
			}
			middle = med3(indexes, bottom, middle, top);
		}
		int partionValue = indexes[middle];
		int a, b, c, d;
		a = b = start;
		c = d = end - 1;
		while(true) {
			while(b <= c && indexes[b] <= partionValue) {
				if(indexes[b] == partionValue) {
					tempIx = indexes[a];
					indexes[a] = indexes[b];
					indexes[b] = tempIx;
					tempVal = values[a];
					values[a++] = values[b];
					values[b] = tempVal;
				}
				b++;
			}
			while(c >= b && indexes[c] >= partionValue) {
				if(indexes[c] == partionValue) {
					tempIx = indexes[c];
					indexes[c] = indexes[d];
					indexes[d] = tempIx;
					tempVal = values[c];
					values[c] = values[d];
					values[d--] = tempVal;
				}
				c--;
			}
			if(b > c) {
				break;
			}
			tempIx = indexes[b];
			indexes[b] = indexes[c];
			indexes[c] = tempIx;
			tempVal = values[b];
			values[b++] = values[c];
			values[c--] = tempVal;
		}
		length = a - start < b - a ? a - start : b - a;
		int l = start;
		int h = b - length;
		while(length-- > 0) {
			tempIx = indexes[l];
			indexes[l] = indexes[h];
			indexes[h] = tempIx;
			tempVal = values[l];
			values[l++] = values[h];
			values[h++] = tempVal;
		}
		length = d - c < end - 1 - d ? d - c : end - 1 - d;
		l = b;
		h = end - length;
		while(length-- > 0) {
			tempIx = indexes[l];
			indexes[l] = indexes[h];
			indexes[h] = tempIx;
			tempVal = values[l];
			values[l++] = values[h];
			values[h++] = tempVal;
		}
		if((length = b - a) > 0) {
			sortByIndex(start, start + length, indexes, values);
		}
		if((length = d - c) > 0) {
			sortByIndex(end - length, end, indexes, values);
		}
	}

	/**
	 * In-place sort of three arrays, only first indexes is used for comparison and second indexes as well as values of
	 * same position are sorted accordingly.
	 * 
	 * @param start    starting index
	 * @param end      ending index
	 * @param indexes  ?
	 * @param indexes2 ?
	 * @param values   ?
	 */
	public static void sortByIndex(int start, int end, int[] indexes, int[] indexes2, double[] values) {
		int tempIx;
		int tempIx2;
		double tempVal;

		int length = end - start;
		if(length < 7) {
			for(int i = start + 1; i < end; i++) {
				for(int j = i; j > start && indexes[j - 1] > indexes[j]; j--) {
					tempIx = indexes[j];
					indexes[j] = indexes[j - 1];
					indexes[j - 1] = tempIx;
					tempIx2 = indexes2[j];
					indexes2[j] = indexes2[j - 1];
					indexes2[j - 1] = tempIx2;
					tempVal = values[j];
					values[j] = values[j - 1];
					values[j - 1] = tempVal;
				}
			}
			return;
		}
		int middle = (start + end) / 2;
		if(length > 7) {
			int bottom = start;
			int top = end - 1;
			if(length > 40) {
				length /= 8;
				bottom = med3(indexes, bottom, bottom + length, bottom + (2 * length));
				middle = med3(indexes, middle - length, middle, middle + length);
				top = med3(indexes, top - (2 * length), top - length, top);
			}
			middle = med3(indexes, bottom, middle, top);
		}
		int partionValue = indexes[middle];
		int a, b, c, d;
		a = b = start;
		c = d = end - 1;
		while(true) {
			while(b <= c && indexes[b] <= partionValue) {
				if(indexes[b] == partionValue) {
					tempIx = indexes[a];
					indexes[a] = indexes[b];
					indexes[b] = tempIx;
					tempIx2 = indexes2[a];
					indexes2[a] = indexes2[b];
					indexes2[b] = tempIx2;
					tempVal = values[a];
					values[a++] = values[b];
					values[b] = tempVal;
				}
				b++;
			}
			while(c >= b && indexes[c] >= partionValue) {
				if(indexes[c] == partionValue) {
					tempIx = indexes[c];
					indexes[c] = indexes[d];
					indexes[d] = tempIx;
					tempIx2 = indexes2[c];
					indexes2[c] = indexes2[d];
					indexes2[d] = tempIx2;
					tempVal = values[c];
					values[c] = values[d];
					values[d--] = tempVal;
				}
				c--;
			}
			if(b > c) {
				break;
			}
			tempIx = indexes[b];
			indexes[b] = indexes[c];
			indexes[c] = tempIx;
			tempIx2 = indexes2[b];
			indexes2[b] = indexes2[c];
			indexes2[c] = tempIx2;
			tempVal = values[b];
			values[b++] = values[c];
			values[c--] = tempVal;
		}
		length = a - start < b - a ? a - start : b - a;
		int l = start;
		int h = b - length;
		while(length-- > 0) {
			tempIx = indexes[l];
			indexes[l] = indexes[h];
			indexes[h] = tempIx;
			tempIx2 = indexes2[l];
			indexes2[l] = indexes2[h];
			indexes2[h] = tempIx2;
			tempVal = values[l];
			values[l++] = values[h];
			values[h++] = tempVal;
		}
		length = d - c < end - 1 - d ? d - c : end - 1 - d;
		l = b;
		h = end - length;
		while(length-- > 0) {
			tempIx = indexes[l];
			indexes[l] = indexes[h];
			indexes[h] = tempIx;
			tempIx2 = indexes2[l];
			indexes2[l] = indexes2[h];
			indexes2[h] = tempIx2;
			tempVal = values[l];
			values[l++] = values[h];
			values[h++] = tempVal;
		}
		if((length = b - a) > 0) {
			sortByIndex(start, start + length, indexes, indexes2, values);
		}
		if((length = d - c) > 0) {
			sortByIndex(end - length, end, indexes, indexes2, values);
		}
	}

	public static void sortByValue(int start, int end, double[] values, int[] indexes) {
		double tempVal;
		int tempIx;

		int length = end - start;
		if(length < 7) {
			for(int i = start + 1; i < end; i++) {
				for(int j = i; j > start && values[j - 1] > values[j]; j--) {
					tempVal = values[j];
					values[j] = values[j - 1];
					values[j - 1] = tempVal;
					tempIx = indexes[j];
					indexes[j] = indexes[j - 1];
					indexes[j - 1] = tempIx;
				}
			}
			return;
		}
		int middle = (start + end) / 2;
		if(length > 7) {
			int bottom = start;
			int top = end - 1;
			if(length > 40) {
				length /= 8;
				bottom = med3(values, bottom, bottom + length, bottom + (2 * length));
				middle = med3(values, middle - length, middle, middle + length);
				top = med3(values, top - (2 * length), top - length, top);
			}
			middle = med3(values, bottom, middle, top);
		}
		double partionValue = values[middle];
		int a, b, c, d;
		a = b = start;
		c = d = end - 1;
		while(true) {
			while(b <= c && values[b] <= partionValue) {
				if(values[b] == partionValue) {
					tempVal = values[a];
					values[a] = values[b];
					values[b] = tempVal;
					tempIx = indexes[a];
					indexes[a++] = indexes[b];
					indexes[b] = tempIx;
				}
				b++;
			}
			while(c >= b && values[c] >= partionValue) {
				if(values[c] == partionValue) {
					tempVal = values[c];
					values[c] = values[d];
					values[d] = tempVal;
					tempIx = indexes[c];
					indexes[c] = indexes[d];
					indexes[d--] = tempIx;
				}
				c--;
			}
			if(b > c) {
				break;
			}
			tempVal = values[b];
			values[b] = values[c];
			values[c] = tempVal;
			tempIx = indexes[b];
			indexes[b++] = indexes[c];
			indexes[c--] = tempIx;
		}
		length = a - start < b - a ? a - start : b - a;
		int l = start;
		int h = b - length;
		while(length-- > 0) {
			tempVal = values[l];
			values[l] = values[h];
			values[h] = tempVal;
			tempIx = indexes[l];
			indexes[l++] = indexes[h];
			indexes[h++] = tempIx;
		}
		length = d - c < end - 1 - d ? d - c : end - 1 - d;
		l = b;
		h = end - length;
		while(length-- > 0) {
			tempVal = values[l];
			values[l] = values[h];
			values[h] = tempVal;
			tempIx = indexes[l];
			indexes[l++] = indexes[h];
			indexes[h++] = tempIx;
		}
		if((length = b - a) > 0) {
			sortByValue(start, start + length, values, indexes);
		}
		if((length = d - c) > 0) {
			sortByValue(end - length, end, values, indexes);
		}
	}

	/**
	 * In-place sort of two arrays, only indexes is used for comparison and values of same position are sorted
	 * accordingly.
	 * 
	 * @param start   start index
	 * @param end     end index
	 * @param values  double array of values to sort
	 * @param indexes int array of indexes to sort by
	 */
	public static void sortByValueStable(int start, int end, double[] values, int[] indexes) {

		sortByValue(start, end, values, indexes);

		// Maintain the stability of the index order.
		for(int i = 0; i < values.length - 1; i++) {
			double tmp = values[i];
			// determine run of equal values
			int len = 0;
			while(i + len + 1 < values.length && tmp == values[i + len + 1])
				len++;
			// unstable sort of run indexes (equal value guaranteed)
			if(len > 0) {
				Arrays.sort(indexes, i, i + len + 1);
				i += len; // skip processed run
			}
		}
	}

	private static int med3(int[] array, int a, int b, int c) {
		int x = array[a], y = array[b], z = array[c];
		return x < y ? (y < z ? b : (x < z ? c : a)) : (y > z ? b : (x > z ? c : a));
	}

	private static int med3(double[] array, int a, int b, int c) {
		double x = array[a], y = array[b], z = array[c];
		return x < y ? (y < z ? b : (x < z ? c : a)) : (y > z ? b : (x > z ? c : a));
	}
}
