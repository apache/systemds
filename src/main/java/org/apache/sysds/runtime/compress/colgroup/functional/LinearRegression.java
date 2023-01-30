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

package org.apache.sysds.runtime.compress.colgroup.functional;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface LinearRegression {

	public static double[] regressMatrixBlock(MatrixBlock rawBlock, IColIndex colIndexes, boolean transposed) {
		final int nRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		final int nCol = colIndexes.size();
		if(nRows <= 1)
			throw new DMLCompressionException("At least 2 data points are required to fit a linear function.");
		else if(nCol < 1)
			throw new DMLCompressionException("At least 1 column must be specified for compression.");

		// the first `nCol` entries represent the intercepts (beta0)
		// the second `nCol` entries represent the slopes (beta1)
		final double[] beta0_beta1 = new double[2 * nCol];

		final double s_xx = (Math.pow(nRows, 3) - nRows) / 12;
		final double x_bar = (double) (nRows + 1) / 2;

		final double[] colSums = new double[nCol];
		final double[] weightedColSums = new double[nCol];

		if(nCol == 1) {
			if(transposed) {
				for(int rowIdx = 0; rowIdx < nRows; rowIdx++) {
					double value = rawBlock.getValue(colIndexes.get(0), rowIdx);
					colSums[0] += value;
					weightedColSums[0] += (rowIdx + 1) * value;
				}
			}
			else {
				for(int rowIdx = 0; rowIdx < nRows; rowIdx++) {
					double value = rawBlock.getValue(rowIdx, colIndexes.get(0));
					colSums[0] += value;
					weightedColSums[0] += (rowIdx + 1) * value;
				}
			}
		}
		else {
			final ReaderColumnSelection reader = ReaderColumnSelection.createReader(rawBlock, colIndexes, transposed);

			DblArray cellVals;
			while((cellVals = reader.nextRow()) != null) {
				int rowIdx = reader.getCurrentRowIndex() + 1;
				double[] row = cellVals.getData();

				for(int i = 0; i < nCol; i++) {
					colSums[i] += row[i];
					weightedColSums[i] += rowIdx * row[i];
				}
			}
		}

		for(int i = 0; i < nCol; i++) {
			beta0_beta1[nCol + i] = (-x_bar * colSums[i] + weightedColSums[i]) / s_xx;
			beta0_beta1[i] = (colSums[i] / nRows) - beta0_beta1[nCol + i] * x_bar;
		}

		return beta0_beta1;
	}
}
