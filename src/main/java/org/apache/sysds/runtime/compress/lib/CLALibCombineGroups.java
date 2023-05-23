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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.IContainDefaultTuple;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.encoding.ConstEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EmptyEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.encoding.SparseEncoding;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Library functions to combine column groups inside a compressed matrix.
 */
public final class CLALibCombineGroups {
	protected static final Log LOG = LogFactory.getLog(CLALibCombineGroups.class.getName());

	private CLALibCombineGroups() {
		// private constructor
	}

	public static CompressedMatrixBlock combine(CompressedMatrixBlock cmb, int k) {
		throw new NotImplementedException();
	}

	/**
	 * Combine the column groups A and B together.
	 * 
	 * The number of rows should be equal, and it is not verified so there will be unexpected behavior in such cases.
	 * 
	 * @param a The first group to combine.
	 * @param b The second group to combine.
	 * @return A new column group containing the two.
	 */
	public static AColGroup combine(AColGroup a, AColGroup b) {
		IColIndex combinedColumns = ColIndexFactory.combine(a, b);

		// try to recompress a and b if uncompressed
		if(a instanceof ColGroupUncompressed)
			a = a.recompress();

		if(b instanceof ColGroupUncompressed)
			b = b.recompress();

		if(a instanceof AColGroupCompressed && b instanceof AColGroupCompressed)
			return combineCompressed(combinedColumns, (AColGroupCompressed) a, (AColGroupCompressed) b);
		else if(a instanceof ColGroupUncompressed || b instanceof ColGroupUncompressed)
			// either side is uncompressed
			return combineUC(combinedColumns, a, b);

		throw new NotImplementedException(
			"Not implemented combine for " + a.getClass().getSimpleName() + " - " + b.getClass().getSimpleName());

	}

	private static AColGroup combineCompressed(IColIndex combinedColumns, AColGroupCompressed ac,
		AColGroupCompressed bc) {
		IEncode ae = ac.getEncoding();
		IEncode be = bc.getEncoding();
		if(ae instanceof SparseEncoding && !(be instanceof SparseEncoding)) {
			// the order must be sparse second unless both sparse.
			return combineCompressed(combinedColumns, bc, ac);
		}

		IEncode ce = ae.combineNoResize(be);

		if(ce instanceof DenseEncoding) {
			DenseEncoding ced = (DenseEncoding) ce;
			ADictionary cd = DictionaryFactory.combineDictionaries(ac, bc);
			return ColGroupDDC.create(combinedColumns, cd, ced.getMap(), null);
		}
		else if(ce instanceof EmptyEncoding) {
			return new ColGroupEmpty(combinedColumns);
		}
		else if(ce instanceof ConstEncoding) {
			ADictionary cd = DictionaryFactory.combineDictionaries(ac, bc);
			return ColGroupConst.create(combinedColumns, cd);
		}
		else if(ce instanceof SparseEncoding) {
			SparseEncoding sed = (SparseEncoding) ce;
			ADictionary cd = DictionaryFactory.combineDictionariesSparse(ac, bc);
			double[] defaultTuple = constructDefaultTuple((AColGroupCompressed) ac, (AColGroupCompressed) bc);
			return ColGroupSDC.create(combinedColumns, sed.getNumRows(), cd, defaultTuple, sed.getOffsets(), sed.getMap(),
				null);
		}

		throw new NotImplementedException(
			"Not implemented combine for " + ac.getClass().getSimpleName() + " - " + bc.getClass().getSimpleName());

	}

	private static AColGroup combineUC(IColIndex combinedColumns, AColGroup a, AColGroup b) {
		int nRow = a instanceof ColGroupUncompressed ? //
			((ColGroupUncompressed) a).getData().getNumRows() : //
			((ColGroupUncompressed) b).getData().getNumRows();
		// step 1 decompress both into target uncompressed MatrixBlock;
		MatrixBlock target = new MatrixBlock(nRow, combinedColumns.size(), false);
		target.allocateBlock();
		DenseBlock db = target.getDenseBlock();

		IColIndex aTempCols = ColIndexFactory.getColumnMapping(combinedColumns, a.getColIndices());
		a.copyAndSet(aTempCols).decompressToDenseBlock(db, 0, nRow, 0, 0);
		IColIndex bTempCols = ColIndexFactory.getColumnMapping(combinedColumns, b.getColIndices());
		b.copyAndSet(bTempCols).decompressToDenseBlock(db, 0, nRow, 0, 0);

		target.recomputeNonZeros();

		return ColGroupUncompressed.create(combinedColumns, target, false);

	}

	public static double[] constructDefaultTuple(AColGroupCompressed ac, AColGroupCompressed bc) {
		double[] ret = new double[ac.getNumCols() + bc.getNumCols()];
		if(ac instanceof IContainDefaultTuple ){
			double[] defa = ((IContainDefaultTuple)ac).getDefaultTuple();
			System.arraycopy(defa, 0, ret, 0, defa.length);
		}
		if(bc instanceof IContainDefaultTuple){
			double[] defb = ((IContainDefaultTuple)bc).getDefaultTuple();
			System.arraycopy(defb, 0, ret, ac.getNumCols(), defb.length);
		}
		return ret;
	}

}
