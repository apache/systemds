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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.fail;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCZeros;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ColGroupMorphingPerformanceCompare {
	protected static final Log LOG = LogFactory.getLog(ColGroupMorphingPerformanceCompare.class.getName());

	@Test
	public void compareMorphedVsNot() {
		run(100, 1, 3, 0.1, 1, 1);
	}

	public static void main(String[] args) {
		// Small scale experiment showing that morphing is better than not morphing.
		for(int i = 0; i < 10; i++)
			run(10000, 1, 100, 0.1, 10, 100);
	}

	public static void run(int nRow, int nCol, int nVal, double redundancy, int repPreAgg, int repAll) {
		MatrixBlock mbt = JolEstimateTest.genRM(nCol, nRow, 1, nVal + 1, redundancy, 2314);

		mbt = mbt.scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1), null);
		CompressionType ct = CompressionType.SDC;
		// we know this is SDC... if it is not the test obviously fail.
		ColGroupSDC g = (ColGroupSDC) ColGroupTest.getColGroup(mbt, ct, mbt.getNumColumns());
		SDCNoMorph gm = new SDCNoMorph(g);

		ColGroupSDCZeros gz = (ColGroupSDCZeros) g.scalarOperation(new RightScalarOperator(Plus.getPlusFnObject(), -1));

		MatrixBlock mbl = JolEstimateTest.genRM(nCol, nRow, 1, nVal + 1, 1.0, 23214); // dense left side.

		MatrixBlock mb = LibMatrixReorg.transpose(mbt);

		MatrixBlock retUncompressed = LibMatrixMult.matrixMult(mbl, mb);

		final MatrixBlock tmpRes = new MatrixBlock(1, retUncompressed.getNumColumns(), false);
		tmpRes.allocateDenseBlock();

		final MatrixBlock ret = new MatrixBlock(retUncompressed.getNumRows(), retUncompressed.getNumColumns(), false);
		ret.allocateDenseBlock();

		final MatrixBlock retNoMorph = new MatrixBlock(retUncompressed.getNumRows(), retUncompressed.getNumColumns(),
			false);
		retNoMorph.allocateDenseBlock();

		try {

			Timing time = new Timing(true);
			// PreAggregate and Morphing
			for(int i = 0; i < repAll; i++) {
				// reset ret.
				ret.reset();

				// morph ... (here simulated)
				final double[] morphSum = new double[] {1};

				double[] preAgg = new double[1];// preAggregate array.
				for(int j = 0; j < repPreAgg; j++) {
					// simulate that we have multiple column groups with same type of values.
					preAgg = new double[nVal];
					gz.preAggregate(mbl, preAgg, 0, 1);
				}

				// multiply preAggregate
				MatrixBlock preAggMB = new MatrixBlock(1, preAgg.length, preAgg);
				gz.mmWithDictionary(preAggMB, tmpRes, ret, 1, 0, 1);

				// row sum left
				MatrixBlock rowSum = mbl.rowSum();

				// multiply row sum
				ColGroupUtils.outerProduct(rowSum.getDenseBlockValues(), morphSum, ret.getDenseBlockValues(), 0, 1);

				// recompute non zeros.
				ret.recomputeNonZeros();
			}
			LOG.info("Time    Morph: " + time.stop());
			time = new Timing(true);
			// PreAggregate no Morphing
			for(int i = 0; i < repAll; i++) {
				// reset ret.
				retNoMorph.reset();

				double[] preAgg = new double[1];// preAggregate array.
				for(int j = 0; j < repPreAgg; j++) {
					// simulate that we have multiple column groups with same type of values.
					preAgg = new double[nVal + 1];
					gm.preAggregate(mbl, preAgg, 0, 1);
				}

				// multiply preAggregate
				MatrixBlock preAggMB = new MatrixBlock(1, preAgg.length, preAgg);
				gm.mmWithDictionary(preAggMB, tmpRes, retNoMorph, 1, 0, 1);

				// No Row Sum needed.

				// recompute non zeros.
				retNoMorph.recomputeNonZeros();
			}
			LOG.info("Time No Morph: " + time.stop());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to run");
		}

		TestUtils.compareMatrices(retUncompressed, ret, 0.001);
		TestUtils.compareMatrices(retUncompressed, retNoMorph, 0.001);

	}

	protected static class SDCNoMorph extends ColGroupSDC {

		private final MatrixBlock mbDict;

		protected SDCNoMorph(int numRows) {
			super(numRows);
			mbDict = null;
		}

		public SDCNoMorph(ColGroupSDC g) {
			this(g.getColIndices(), g.getNumRows(), g.getDictionary(), g.getDefaultTuple(), g.getOffsets(), g.getMapping(),
				null);
		}

		protected SDCNoMorph(int[] colIndices, int numRows, ADictionary dict, double[] defaultTuple, AOffset offsets,
			AMapToData data, int[] cachedCounts) {
			super(colIndices, numRows, dict, defaultTuple, offsets, data, cachedCounts);

			MatrixBlock tmp = getDictionary().getMBDict(_colIndexes.length).getMatrixBlock();
			mbDict = tmp.append(new MatrixBlock(1, defaultTuple.length, defaultTuple), false);

		}

		/**
		 * Pre aggregate a matrix block into a pre aggregate target (first step of left matrix multiplication)
		 * 
		 * @param m      The matrix to preAggregate
		 * @param preAgg The preAggregate target
		 * @param rl     Row lower on the left side matrix
		 * @param ru     Row upper on the left side matrix
		 */
		public final void preAggregate(MatrixBlock m, double[] preAgg, int rl, int ru) {
			if(m.isInSparseFormat())
				preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
			else
				preAggregateDense(m, preAgg, rl, ru, 0, m.getNumColumns());
		}

		private final void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {
			// throw new NotImplementedException();
			AIterator it = _indexes.getIterator();

			final double[] vals = m.getDenseBlockValues();
			final int last = _indexes.getOffsetToLast();

			int c = 0;
			for(; c < cu && it.value() < last; c++) {
				if(it.value() == c) {
					preAgg[_data.getIndex(it.getDataIndex())] += vals[c];
					it.next();
				}
				else
					preAgg[preAgg.length - 1] += vals[c];

			}

			for(; c < last; c++) // until last
				preAgg[preAgg.length - 1] += vals[c];

			// add last
			preAgg[_data.getIndex(it.getDataIndex())] += vals[c++];

			for(; c < cu; c++) // add to default
				preAgg[preAgg.length - 1] += vals[c];

		}

		private final void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru) {
			throw new NotImplementedException();
		}

		public void mmWithDictionary(MatrixBlock preAgg, MatrixBlock tmpRes, MatrixBlock ret, int k, int rl, int ru) {
			// Shallow copy the preAgg to allow sparse PreAgg multiplication but do not remove the original dense
			// allocation
			// since the dense allocation is reused.
			final MatrixBlock preAggCopy = new MatrixBlock();
			preAggCopy.copy(preAgg);
			final MatrixBlock tmpResCopy = new MatrixBlock();
			tmpResCopy.copy(tmpRes);
			// Get dictionary matrixBlock
			// final MatrixBlock dict = getDictionary().getMBDict(_colIndexes.length).getMatrixBlock();
			// if(dict != null) {
			// Multiply
			LibMatrixMult.matrixMult(preAggCopy, mbDict, tmpResCopy, k);
			ColGroupUtils.addMatrixToResult(tmpResCopy, ret, _colIndexes, rl, ru);
			// }
		}
	}

}
