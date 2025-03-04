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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import java.io.DataInput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.ColIndexType;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public interface ColIndexFactory {
	public static final Log LOG = LogFactory.getLog(ColIndexFactory.class.getName());

	public static IColIndex read(DataInput in) throws IOException {
		final ColIndexType t = ColIndexType.values()[in.readByte()];
		switch(t) {
			case SINGLE:
				return new SingleIndex(in.readInt());
			case TWO:
				return new TwoIndex(in.readInt(), in.readInt());
			case ARRAY:
				return ArrayIndex.read(in);
			case RANGE:
				return RangeIndex.read(in);
			case TWORANGE: 
				return TwoRangesIndex.read(in);
			case COMBINED:
				return CombinedIndex.read(in);
			default:
				throw new DMLCompressionException("Failed reading column index of type: " + t);
		}
	}

	public static IColIndex createI(int... indexes) {
		return create(indexes);
	}

	public static IColIndex create(int[] indexes) {
		if(indexes.length <= 0)
			throw new DMLRuntimeException("Invalid length to create index from : " + indexes.length);
		else if(indexes.length == 1)
			return new SingleIndex(indexes[0]);
		else if(indexes.length == 2)
			return new TwoIndex(indexes[0], indexes[1]);
		else if(RangeIndex.isValidRange(indexes))
			return new RangeIndex(indexes[0], indexes[0] + indexes.length);
		else
			return new ArrayIndex(indexes);
	}

	public static IColIndex create(IntArrayList indexes) {
		final int s = indexes.size();
		if(s <= 0)
			throw new DMLRuntimeException("Invalid length to create index from " + indexes);
		else if(s == 1)
			return new SingleIndex(indexes.get(0));
		else if(s == 2)
			return new TwoIndex(indexes.get(0), indexes.get(1));
		else if(RangeIndex.isValidRange(indexes))
			return new RangeIndex(indexes.get(0), indexes.get(0) + s);
		else
			return new ArrayIndex(indexes.extractValues(true));
	}

	/**
	 * Create an Index range of the given values
	 * 
	 * @param l Lower bound (inclusive)
	 * @param u Upper bound (not inclusive)
	 * @return An Index
	 */
	public static IColIndex create(int l, int u) {
		if(u - l <= 0)
			throw new DMLRuntimeException("Invalid range: " + l + " " + u);
		else if(u - 1 == l)
			return new SingleIndex(l);
		else if(u - 2 == l)
			return new TwoIndex(l, l + 1);
		else
			return new RangeIndex(l, u);
	}

	public static IColIndex create(int nCol) {
		if(nCol <= 0)
			throw new DMLRuntimeException("Invalid size of index columns must be above 0");
		else if(nCol == 1)
			return new SingleIndex(0);
		else if(nCol == 2)
			return new TwoIndex(0, 1);
		else
			return new RangeIndex(nCol);
	}

	public static long estimateMemoryCost(int nCol, boolean contiguous) {
		if(nCol == 1)
			return SingleIndex.estimateInMemorySizeStatic();
		else if(nCol == 2)
			return TwoIndex.estimateInMemorySizeStatic();
		else if(contiguous)
			return RangeIndex.estimateInMemorySizeStatic();
		else
			return ArrayIndex.estimateInMemorySizeStatic(nCol);
	}

	public static IColIndex combine(List<? extends AColGroup> gs) {
		int numCols = 0;
		for(AColGroup g : gs)
			numCols += g.getNumCols();

		int[] resCols = new int[numCols];

		int index = 0;
		for(AColGroup g : gs) {
			final IIterate it = g.getColIndices().iterator();
			while(it.hasNext())
				resCols[index++] = it.next();
		}

		Arrays.sort(resCols);
		return create(resCols);
	}

	public static IColIndex combineIndexes(List<IColIndex> idx) {
		int numCols = 0;
		for(IColIndex g : idx)
			numCols += g.size();

		int[] resCols = new int[numCols];

		int index = 0;
		for(IColIndex g : idx) {
			final IIterate it = g.iterator();
			while(it.hasNext())
				resCols[index++] = it.next();
		}

		Arrays.sort(resCols);
		return create(resCols);
	}

	public static IColIndex combine(AColGroup a, AColGroup b) {
		return combine(a.getColIndices(), b.getColIndices());
	}

	public static IColIndex combine(IColIndex a, IColIndex b) {
		final int numCols = a.size() + b.size();
		final int[] resCols = new int[numCols];
		int index = 0;
		final IIterate ita = a.iterator();
		while(ita.hasNext())
			resCols[index++] = ita.next();
		final IIterate itb = b.iterator();
		while(itb.hasNext())
			resCols[index++] = itb.next();
		Arrays.sort(resCols);
		return create(resCols);
	}

	/**
	 * Provide a mapping from a to the combined columns shifted over to column positions in the combined.
	 * 
	 * It is assumed that the caller always input an a that is contained in comb. it is not verified in the call that it
	 * is correct.
	 * 
	 * @param comb The combined indexes
	 * @param a    The indexes to look up
	 * @return A column index mapping.
	 */
	public static IColIndex getColumnMapping(IColIndex comb, IColIndex a) {

		// naive scan entire a, there are options that make this faster depending on what comb look like
		// but this is most commonly not relevant.
		final int numCols = a.size();
		final int[] ret = new int[numCols];
		final IIterate itc = comb.iterator();
		final IIterate ita = a.iterator();
		int index = 0;
		while(ita.hasNext()) {
			while(itc.v() < ita.v())
				itc.next();
			ret[index++] = itc.i();
			ita.next();
		}

		return create(ret);
	}

}
