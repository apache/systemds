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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PiecewiseLinearUtils {

	private PiecewiseLinearUtils() {
	}

	public static final class SegmentedRegression {
		private final int[] breakpoints;
		private final double[] slopes;
		private final double[] intercepts;

		public SegmentedRegression(int[] breakpoints, double[] slopes, double[] intercepts) {
			this.breakpoints = breakpoints;
			this.slopes = slopes;
			this.intercepts = intercepts;
		}

		public int[] getBreakpoints() {
			return breakpoints;
		}

		public double[] getSlopes() {
			return slopes;
		}

		public double[] getIntercepts() {
			return intercepts;
		}
	}

	public static double[] getColumn(MatrixBlock in, int colIndex) {
		final int numRows = in.getNumRows();
		final double[] column = new double[numRows];
		for(int row = 0; row < numRows; row++) {
			column[row] = in.get(row, colIndex);
		}
		return column;
	}

	public static double[] getRow(MatrixBlock in, int rowIndex) {
		final int numCols = in.getNumColumns();
		final double[] row = new double[numCols];
		for(int col = 0; col < numCols; col++) {
			row[col] = in.get(rowIndex, col);
		}
		return row;
	}

	/**
	 * Compresses a single column into a piecewise-linear model using a successive breakpoint algorithm.
	 *
	 * @param column the input data column
	 * @param cs     compression settings controlling the allowed per-segment approximation error
	 * @return a segmented regression representation of the column
	 */
	public static SegmentedRegression compressSuccessivePiecewiseLinear(double[] column, CompressionSettings cs) {

		final List<Integer> breakpointsList = computeBreakpointSuccessive(column, cs);
		final int[] breakpoints = breakpointsList.stream().mapToInt(Integer::intValue).toArray();

		final int numSeg = breakpoints.length - 1;
		final double[] slopes = new double[numSeg];
		final double[] intercepts = new double[numSeg];

		for(int seg = 0; seg < numSeg; seg++) {
			final int segstart = breakpoints[seg];
			final int segEnd = breakpoints[seg + 1];
			final double[] line = regressSegment(column, segstart, segEnd);
			slopes[seg] = line[0];
			intercepts[seg] = line[1];
		}
		return new SegmentedRegression(breakpoints, slopes, intercepts);
	}

	public static double[] regressSegment(double[] column, int start, int end) {
		final int numElements = end - start;
		if(numElements <= 0)
			return new double[] {0.0, 0.0};

		double sumOfRowIndices = 0, sumOfColumnValues = 0, sumOfRowIndicesSquared = 0,
			productRowIndexTimesColumnValue = 0;
		for(int i = start; i < end; i++) {
			sumOfRowIndices += i;
			sumOfColumnValues += column[i];
			sumOfRowIndicesSquared += i * i;
			productRowIndexTimesColumnValue += i * column[i];
		}

		final double denominatorForSlope = numElements * sumOfRowIndicesSquared - sumOfRowIndices * sumOfRowIndices;
		final double slope;
		final double intercept;
		if(denominatorForSlope == 0) {
			slope = 0.0;
			intercept = sumOfColumnValues / numElements;
		}
		else {
			slope = (numElements * productRowIndexTimesColumnValue - sumOfRowIndices * sumOfColumnValues) /
				denominatorForSlope;
			intercept = (sumOfColumnValues - slope * sumOfRowIndices) / numElements;
		}
		return new double[] {slope, intercept};
	}

	/**
	 * computes breakpoints for a y using a successive algorithm extends each segment until the SEE reaches the target
	 * loss, then start a new segment
	 *
	 * @param y  y values
	 * @param cs compression setting for setting the target loss
	 * @return list of breakpoint indices
	 */
	public static List<Integer> computeBreakpointSuccessive(double[] y, CompressionSettings cs) {
		final int numElements = y.length;
		final double targetMSE = cs.getPiecewiseTargetLoss();
		if(Double.isNaN(targetMSE) || targetMSE <= 0) {
			return Arrays.asList(0, numElements); // fallback single segment
		}

		List<Integer> breakpoints = new ArrayList<>();
		breakpoints.add(0);
		double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
		int segmentLength = 0;
		double beta, alpha;

		for(int n = 0; n < numElements; n++) {
			double x = n;
			double Y = y[n];
			sumX += x;
			sumY += Y;
			sumX2 += x * x;
			sumY2 += Y * Y;
			sumXY += x * Y;
			segmentLength++;

			if(segmentLength > 1) {

				final double alphaBetaDenominator = segmentLength * sumX2 - sumX * sumX;
				if(alphaBetaDenominator == 0.0) {
					beta = 0.0;
					alpha = sumY / segmentLength;
				}
				else {
					beta = (segmentLength * sumXY - sumX * sumY) / alphaBetaDenominator;
					alpha = (sumY * sumX2 - sumX * sumXY) / alphaBetaDenominator;
				}

				double sse = Math.max(0.0, sumY2 - alpha * sumY - beta * sumXY); // sum of least squares
				if(sse > segmentLength * targetMSE) {
					breakpoints.add(n);
					segmentLength = 1;
					sumX = x;
					sumY = Y;
					sumX2 = x * x;
					sumY2 = Y * Y;
					sumXY = x * Y;
				}
			}
		}

		// make sure, that the last breakpoint equals numElements
		int last = breakpoints.get(breakpoints.size() - 1);
		if(last != numElements) {
			breakpoints.add(numElements);
		}

		return breakpoints;
	}
}
