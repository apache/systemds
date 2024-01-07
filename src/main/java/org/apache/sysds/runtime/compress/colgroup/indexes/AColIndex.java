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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;

public abstract class AColIndex implements IColIndex {

	protected static final Log LOG = LogFactory.getLog(AColIndex.class.getName());

	@Override
	public int hashCode() {
		return hashCode(iterator());
	}

	@Override
	public boolean equals(Object other) {
		return other instanceof IColIndex && this.equals((IColIndex) other);
	}

	@Override
	public boolean contains(IColIndex a, IColIndex b) {
		return a != null && b != null && findIndex(a.get(0)) >= 0 && findIndex(b.get(0)) >= 0;
	}

	@Override
	public boolean containsStrict(IColIndex a, IColIndex b) {
		if(a != null && b != null && a.size() + b.size() == size()) {
			IIterate ia = a.iterator();
			while(ia.hasNext()) {
				if(!(findIndex(ia.next()) >= 0))
					return false;
			}

			IIterate ib = b.iterator();
			while(ib.hasNext()) {
				if(!(findIndex(ib.next()) >= 0))
					return false;
			}
			return true;
		}
		return false;
	}

	private static int hashCode(IIterate it) {
		int res = 1;
		while(it.hasNext())
			res = 31 * res + it.next();
		return res;
	}

	@Override
	public boolean containsAny(IColIndex idx) {
		if(idx instanceof TwoRangesIndex){
			TwoRangesIndex o = (TwoRangesIndex) idx;
			return this.containsAny(o.idx1) || this.containsAny(o.idx2);
		}
		else if(idx instanceof CombinedIndex){
			CombinedIndex ci = (CombinedIndex) idx;
			return containsAny(ci.l) || containsAny(ci.r);
		}
		else{
			final IIterate it = idx.iterator();
			while(it.hasNext())
				if(contains(it.next()))
					return true;
	
			return false;
		}
	}

	@Override
	public void decompressToDenseFromSparse(SparseBlock sb, int vr, int off, double[] c) {
		if(sb instanceof SparseBlockCSR)
			decompressToDenseFromSparseCSR((SparseBlockCSR)sb, vr, off, c);
		else
			decompressToDenseFromSparseGeneric(sb, vr, off, c);
	}

	private void decompressToDenseFromSparseGeneric(SparseBlock sb, int vr, int off, double[] c) {
		if(sb.isEmpty(vr))
			return;
		final int apos = sb.pos(vr);
		final int alen = sb.size(vr) + apos;
		final int[] aix = sb.indexes(vr);
		final double[] aval = sb.values(vr);
		for(int j = apos; j < alen; j++)
			c[off + get(aix[j])] += aval[j];
	}

	private void decompressToDenseFromSparseCSR(SparseBlockCSR sb, int vr, int off, double[] c) {
		final int apos = sb.pos(vr);
		final int alen = sb.size(vr) + apos;
		final int[] aix = sb.indexes(vr);
		final double[] aval = sb.values(vr);
		for(int j = apos; j < alen; j++)
			c[off + get(aix[j])] += aval[j];
	}

	@Override
	public void decompressVec(int nCol, double[] c, int off, double[] values, int rowIdx) {
		for(int j = 0; j < nCol; j++)
			c[off + get(j)] += values[rowIdx + j];
	}
}
