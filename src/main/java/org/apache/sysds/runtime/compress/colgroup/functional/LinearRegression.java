package org.apache.sysds.runtime.compress.colgroup.functional;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class LinearRegression {
	public static double[][] regressMatrixBlock(MatrixBlock rawBlock, int[] colIndices, boolean transposed) {
		final int n = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();

		if(n <= 1) {
			throw new DMLCompressionException("At least 2 data points are required to fit a linear function.");
		}

		final int nCols = transposed ? rawBlock.getNumRows() : rawBlock.getNumColumns();
		double[] beta0 = new double[nCols];
		double[] beta1 = new double[nCols];

		double s_xx = (Math.pow(n, 3) - n) / 12;
		double x_bar = (double) (n + 1) / 2;

		ReaderColumnSelection reader = ReaderColumnSelection.createReader(rawBlock, colIndices, transposed);
		MatrixBlock colSums = transposed ? rawBlock.rowSum() : rawBlock.colSum();
		double[] weightedColSums = new double[nCols];

		DblArray cellVals = null;
		while((cellVals = reader.nextRow()) != null) {
			int rowIdx = reader.getCurrentRowIndex() + 1;
			double[] row = cellVals.getData();

			for(int i = 0; i < nCols; i++) {
				weightedColSums[i] += rowIdx * row[i];
			}
		}

		for(int i = 0; i < nCols; i++) {
			double colSumValue = transposed ? colSums.getValue(i, 0) : colSums.getValue(0, i);

			beta1[i] = (-x_bar * colSumValue + weightedColSums[i]) / s_xx;
			beta0[i] = (colSumValue / n) - beta1[i] * x_bar;
		}

		return new double[][] {beta0, beta1};
	}
}
