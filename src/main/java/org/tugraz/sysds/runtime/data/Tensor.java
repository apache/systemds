/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.data;

import org.tugraz.sysds.runtime.controlprogram.caching.CacheBlock;
import org.tugraz.sysds.runtime.util.UtilFunctions;

public abstract class Tensor implements CacheBlock
{
	public static final int[] DEFAULT_DIMS = new int[]{0, 0};

	//min 2 dimensions to preserve proper matrix semantics
	protected int[] _dims; //[2,inf)

	public abstract void reset();

	public abstract void reset(int[] dims);

	public abstract boolean isAllocated();

	public abstract Tensor allocateBlock();

	public int getNumDims() {
		return _dims.length;
	}

	public int getNumRows() {
		return getDim(0);
	}

	public int getNumColumns() {
		return getDim(1);
	}

	public int getDim(int i) {
		return _dims[i];
	}

	/**
	 * Calculates the next index array. Note that if the given index array was the last element, the next index will
	 * be the first one.
	 *
	 * @param ix the index array which will be incremented to the next index array
	 */
	public void getNextIndexes(int[] ix) {
		int i = ix.length - 1;
		ix[i]++;
		//calculating next index
		if (ix[i] == getDim(i)) {
			while (ix[i] == getDim(i)) {
				ix[i] = 0;
				i--;
				if (i < 0) {
					//we are finished
					break;
				}
				ix[i]++;
			}
		}
	}

	public boolean isVector() {
		return getNumDims() <= 2
				&& (getDim(0) == 1 || getDim(1) == 1);
	}

	public boolean isMatrix() {
		return getNumDims() == 2
				&& (getDim(0) > 1 && getDim(1) > 1);
	}

	public long getLength() {
		return UtilFunctions.prod(_dims);
	}

	public boolean isEmpty() {
		return isEmpty(false);
	}

	public abstract boolean isEmpty(boolean safe);

	public abstract double get(int[] ix);

	public abstract double get(int r, int c);

	public abstract long getLong(int[] ix);

	public abstract String getString(int[] ix);

	public abstract void set(int[] ix, double v);

	public abstract void set(int r, int c, double v);

	public abstract void set(int[] ix, long v);

	public abstract void set(int[] ix, String v);
}
