package org.apache.sysds.runtime.compress.colgroup.functional;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class LinearRegression {
	public static double[][] regressMatrixBlock(MatrixBlock rawBlock, int[] colIndexes, boolean transposed) {
		final int nRows = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		if(nRows <= 1) {
			throw new DMLCompressionException("At least 2 data points are required to fit a linear function.");
		}

		if(colIndexes.length < 1) {
			throw new DMLCompressionException("At least 1 column must be specified for compression.");
		}

//		final int nCols = transposed ? rawBlock.getNumRows() : rawBlock.getNumColumns();
		double[] beta0 = new double[colIndexes.length];
		double[] beta1 = new double[colIndexes.length];

		double s_xx = (Math.pow(nRows, 3) - nRows) / 12;
		double x_bar = (double) (nRows + 1) / 2;

		MatrixBlock colSums = transposed ? rawBlock.rowSum() : rawBlock.colSum();
		double[] weightedColSums = new double[colIndexes.length];

		if(colIndexes.length == 1) {
			for (int rowIdx = 0; rowIdx < nRows; rowIdx++) {
				double value = transposed ? rawBlock.getValue(colIndexes[0], rowIdx) : rawBlock.getValue(rowIdx, colIndexes[0]);
				weightedColSums[0] += (rowIdx + 1) * value;
			}
		} else {
			ReaderColumnSelection reader = ReaderColumnSelection.createReader(rawBlock, colIndexes, transposed);

			DblArray cellVals;
			while((cellVals = reader.nextRow()) != null) {
				int rowIdx = reader.getCurrentRowIndex() + 1;
				double[] row = cellVals.getData();

				for(int i = 0; i < colIndexes.length; i++) {
					weightedColSums[i] += rowIdx * row[colIndexes[i]];
				}
			}
		}

		for(int i = 0; i < colIndexes.length; i++) {
			double colSumValue = transposed ? colSums.getValue(i, 0) : colSums.getValue(0, i);

			beta1[i] = (-x_bar * colSumValue + weightedColSums[i]) / s_xx;
			beta0[i] = (colSumValue / nRows) - beta1[i] * x_bar;
		}

		return new double[][] {beta0, beta1};
	}
}
