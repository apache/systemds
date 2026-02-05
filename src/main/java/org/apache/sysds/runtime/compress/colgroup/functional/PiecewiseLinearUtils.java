package org.apache.sysds.runtime.compress.colgroup.functional;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

	public static SegmentedRegression compressSegmentedLeastSquares(double[] column, CompressionSettings cs) {
		//compute Breakpoints for a Column with dynamic Programming
		final List<Integer> breakpointsList = computeBreakpoints(cs, column);
		final int[] breakpoints = breakpointsList.stream().mapToInt(Integer::intValue).toArray();

		//get values for Regression
		final int numSeg = breakpoints.length - 1;
		final double[] slopes = new double[numSeg];
		final double[] intercepts = new double[numSeg];

		// Regress per Segment
		for (int seg = 0; seg < numSeg; seg++) {
			final int SegStart = breakpoints[seg];
			final int SegEnd = breakpoints[seg + 1];

			final double[] line = regressSegment(column, SegStart, SegEnd);
			slopes[seg] = line[0]; //slope regession line
			intercepts[seg] = line[1]; //intercept regression line
		}

		return new SegmentedRegression(breakpoints, slopes, intercepts);
	}

	public static  SegmentedRegression compressSegmentedLeastSquaresV2(double[] column, CompressionSettings cs) {
		//compute Breakpoints for a Column with Greedy Algorithm

		final List<Integer> breakpointsList = computeBreakpointsGreedy(column, cs);
		final int[] breakpoints = breakpointsList.stream().mapToInt(Integer::intValue).toArray();

		//get values for Regression
		final int numSeg = breakpoints.length - 1;
		final double[] slopes = new double[numSeg];
		final double[] intercepts = new double[numSeg];

		// Regress per Segment
		for (int seg = 0; seg < numSeg; seg++) {
			final int segstart = breakpoints[seg];
			final int segEnd = breakpoints[seg + 1];
			final double[] line = regressSegment(column, segstart, segEnd);
			slopes[seg] = line[0];
			intercepts[seg] = line[1];
		}
		return new SegmentedRegression(breakpoints,slopes, intercepts);
	}

	public static double[] getColumn(MatrixBlock in, int colIndex) {
		final int numRows = in.getNumRows();
		final double[] column = new double[numRows];

		for (int row = 0; row < numRows; row++) {
			column[row] = in.get(row, colIndex);
		}
		return column;
	}

	public static List<Integer> computeBreakpoints(CompressionSettings cs, double[] column) {
		final int numElements = column.length;
		final double targetMSE = cs.getPiecewiseTargetLoss();


		// TODO: Maybe remove Fallback if no targetloss is given
		/*if (Double.isNaN(targetMSE) || targetMSE <= 0) {
			final double segmentPenalty = 2.0 * Math.log(numElements);
			return computeBreakpointsLambda(column, segmentPenalty);
		}*/

		// max targetloss
		final double sseMax = numElements * targetMSE;
		double minLoss = 0.0;
		double maxLoss = numElements * 100.0;
		List<Integer> bestBreaks = null;
		//compute breakpoints
		while(maxLoss -minLoss > 1e-8) {
			final double currentLoss = 0.5 * (minLoss + maxLoss);
			final List<Integer> breaks = computeBreakpointsLambda(column, currentLoss);
			final double totalSSE = computeTotalSSE(column, breaks);
			if (totalSSE <= sseMax) {
				bestBreaks = breaks;
				minLoss = currentLoss;
			}
			else {
				maxLoss = currentLoss;
			}
		}

		if (bestBreaks == null)
			bestBreaks = computeBreakpointsLambda(column, minLoss);

		return bestBreaks;
	}

	public static List<Integer> computeBreakpointsLambda(double[] column, double lambda) {
		final int numrows = column.length;
		final double[] costs = new double[numrows + 1]; //min Cost
		final int[] prevStart = new int[numrows + 1]; //previous Start
		costs[0] = 0.0;
		// Find Cost
		for (int rowEnd = 1; rowEnd <= numrows; rowEnd++) {
			costs[rowEnd] = Double.POSITIVE_INFINITY;
			//Test all possible Segment to find the lowest costs
			for (int rowStart = 0; rowStart < rowEnd; rowStart++) {
				//costs = current costs + segmentloss + penaltiy
				final double costCurrentSegment = computeSegmentCost(column, rowStart, rowEnd);
				final double totalCost = costs[rowStart] + costCurrentSegment + lambda;
				// Check if it is the better solution
				if (totalCost < costs[rowEnd]) {
					costs[rowEnd] = totalCost;
					prevStart[rowEnd] = rowStart;
				}
			}
		}
		//Check the optimal segmentlimits
		final List<Integer> segmentLimits = new ArrayList<>();
		int breakpointIndex = numrows;
		while (breakpointIndex > 0) {
			segmentLimits.add(breakpointIndex);
			breakpointIndex = prevStart[breakpointIndex];
		}
		segmentLimits.add(0);
		Collections.sort(segmentLimits);
		return segmentLimits;
	}

	public static double computeSegmentCost(double[] column, int start, int end) {
		final int segSize = end - start;
		if (segSize <= 1)
			return 0.0;

		final double[] ab = regressSegment(column, start, end); //Regressionline
		final double slope = ab[0];
		final double intercept = ab[1];

		double sumSquaredError = 0.0;
		for (int i = start; i < end; i++) {
			final double rowIdx = i;
			final double actualValue = column[i];
			final double predictedValue = slope * rowIdx + intercept;
			final double difference = actualValue - predictedValue;
			sumSquaredError += difference * difference;
		}
		return sumSquaredError;
	}

	public static double computeTotalSSE(double[] column, List<Integer> breaks) {
		double total = 0.0;
		for (int s = 0; s < breaks.size() - 1; s++) {
			final int start = breaks.get(s);
			final int end = breaks.get(s + 1);
			total += computeSegmentCost(column, start, end);
		}
		return total;
	}

	public static double[] regressSegment(double[] column, int start, int end) {
		final int numElements = end - start;
		if (numElements <= 0)
			return new double[] {0.0, 0.0};

		double sumOfRowIndices = 0, sumOfColumnValues = 0, sumOfRowIndicesSquared = 0, productRowIndexTimesColumnValue = 0;
		for (int i = start; i < end; i++) {
			final double x = i;
			final double y = column[i];
			sumOfRowIndices += x;
			sumOfColumnValues += y;
			sumOfRowIndicesSquared += x * x;
			productRowIndexTimesColumnValue += x * y;
		}

		final double numPointsInSegmentDouble = numElements;
		final double denominatorForSlope = numPointsInSegmentDouble * sumOfRowIndicesSquared - sumOfRowIndices * sumOfRowIndices;
		final double slope;
		final double intercept;
		if (denominatorForSlope == 0) {
			slope = 0.0;
			intercept = sumOfColumnValues / numPointsInSegmentDouble;
		}
		else {
			slope = (numPointsInSegmentDouble * productRowIndexTimesColumnValue - sumOfRowIndices * sumOfColumnValues) / denominatorForSlope;
			intercept = (sumOfColumnValues - slope * sumOfRowIndices) / numPointsInSegmentDouble;
		}
		return new double[] {slope, intercept};
	}
	public static List<Integer> computeBreakpointsGreedy(double[] column, CompressionSettings cs) {
		final int numElements = column.length;
		final double targetMSE = cs.getPiecewiseTargetLoss();
		if (Double.isNaN(targetMSE) || targetMSE <= 0) {
			return Arrays.asList(0, numElements);  // Fallback: ein Segment
		}

		List<Integer> breakpoints = new ArrayList<>();
		breakpoints.add(0);
		int currentStart = 0;

		while (currentStart < numElements) {
			int bestEnd = numElements;  // Default: Rest als Segment
			for (int end = currentStart + 1; end <= numElements; end++) {
				double sse = computeSegmentCost(column, currentStart, end);
				double sseMax = (end - currentStart) * targetMSE;
				if (sse > sseMax) {
					bestEnd = end - 1;  // Letzter gültiger Endpunkt
					break;
				}
			}
			breakpoints.add(bestEnd);
			currentStart = bestEnd;
		}

		if (breakpoints.get(breakpoints.size() - 1) != numElements) {
			breakpoints.add(numElements);
		}
		return breakpoints;
	}
}
