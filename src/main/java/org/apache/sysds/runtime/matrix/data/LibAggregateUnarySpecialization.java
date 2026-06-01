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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

public class LibAggregateUnarySpecialization {
	protected static final Log LOG = LogFactory.getLog(LibAggregateUnarySpecialization.class.getName());

	public static void aggregateUnary(final MatrixBlock mb, AggregateUnaryOperator op, MatrixBlock result, int blen,
		MatrixIndexes indexesIn) {
		if(op.sparseSafe)
			sparseAggregateUnaryHelp(mb, op, result, blen, indexesIn);
		else
			denseAggregateUnaryHelp(mb, op, result, blen, indexesIn);
	}

	private static void sparseAggregateUnaryHelp(final MatrixBlock mb, AggregateUnaryOperator op, MatrixBlock result,
		int blen, MatrixIndexes indexesIn) {
		// initialize result
		if(op.aggOp.initialValue != 0)
			result.reset(result.rlen, result.clen, op.aggOp.initialValue);
		CellIndex tempCellIndex = new CellIndex(-1, -1);
		KahanObject buffer = new KahanObject(0, 0);

		if(mb.sparse && mb.sparseBlock != null) {
			SparseBlock a = mb.sparseBlock;
			for(int r = 0; r < Math.min(mb.rlen, a.numRows()); r++) {
				if(a.isEmpty(r))
					continue;
				int apos = a.pos(r);
				int alen = a.size(r);
				int[] aix = a.indexes(r);
				double[] aval = a.values(r);
				for(int i = apos; i < apos + alen; i++) {
					tempCellIndex.set(r, aix[i]);
					op.indexFn.execute(tempCellIndex, tempCellIndex);
					incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column, aval[i],
						buffer);
				}
			}
		}
		else if(!mb.sparse && mb.denseBlock != null) {
			DenseBlock a = mb.getDenseBlock();
			for(int i = 0; i < mb.rlen; i++)
				for(int j = 0; j < mb.clen; j++) {
					tempCellIndex.set(i, j);
					op.indexFn.execute(tempCellIndex, tempCellIndex);
					incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column, a.get(i, j),
						buffer);
				}
		}
	}

	private static void denseAggregateUnaryHelp(MatrixBlock mb, AggregateUnaryOperator op, MatrixBlock result, int blen,
		MatrixIndexes indexesIn) {
		if(op.aggOp.initialValue != 0)
			result.reset(result.rlen, result.clen, op.aggOp.initialValue);
		CellIndex tempCellIndex = new CellIndex(-1, -1);
		KahanObject buffer = new KahanObject(0, 0);
		for(int i = 0; i < mb.rlen; i++)
			for(int j = 0; j < mb.clen; j++) {
				tempCellIndex.set(i, j);
				op.indexFn.execute(tempCellIndex, tempCellIndex);
				incrementalAggregateUnaryHelp(op.aggOp, result, tempCellIndex.row, tempCellIndex.column,
					mb.get(i, j), buffer);
			}
	}

	private static void incrementalAggregateUnaryHelp(AggregateOperator aggOp, MatrixBlock result, int row, int column,
		double newvalue, KahanObject buffer) {
		if(aggOp.existsCorrection()) {
			if(aggOp.correction == CorrectionLocationType.LASTROW ||
				aggOp.correction == CorrectionLocationType.LASTCOLUMN) {
				int corRow = row, corCol = column;
				if(aggOp.correction == CorrectionLocationType.LASTROW)// extra row
					corRow++;
				else if(aggOp.correction == CorrectionLocationType.LASTCOLUMN)
					corCol++;
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: " + aggOp.correction);

				buffer._sum = result.get(row, column);
				buffer._correction = result.get(corRow, corCol);
				buffer = (KahanObject) aggOp.increOp.fn.execute(buffer, newvalue);
				result.set(row, column, buffer._sum);
				result.set(corRow, corCol, buffer._correction);
			}
			else if(aggOp.correction == CorrectionLocationType.NONE) {
				throw new DMLRuntimeException("unrecognized correctionLocation: " + aggOp.correction);
			}
			else// for mean
			{
				int corRow = row, corCol = column;
				int countRow = row, countCol = column;
				if(aggOp.correction == CorrectionLocationType.LASTTWOROWS) {
					countRow++;
					corRow += 2;
				}
				else if(aggOp.correction == CorrectionLocationType.LASTTWOCOLUMNS) {
					countCol++;
					corCol += 2;
				}
				else
					throw new DMLRuntimeException("unrecognized correctionLocation: " + aggOp.correction);
				buffer._sum = result.get(row, column);
				buffer._correction = result.get(corRow, corCol);
				double count = result.get(countRow, countCol) + 1.0;
				buffer = (KahanObject) aggOp.increOp.fn.execute(buffer, newvalue, count);
				result.set(row, column, buffer._sum);
				result.set(corRow, corCol, buffer._correction);
				result.set(countRow, countCol, count);
			}

		}
		else {
			newvalue = aggOp.increOp.fn.execute(result.get(row, column), newvalue);
			result.set(row, column, newvalue);
		}
	}

}
