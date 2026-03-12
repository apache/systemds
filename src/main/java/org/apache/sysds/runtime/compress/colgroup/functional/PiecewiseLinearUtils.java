package org.apache.sysds.runtime.compress.colgroup.functional;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class PiecewiseLinearUtils {
	/**
	 * Utility methods for piecewise linear compression of matric columns
	 * supports compression used the segmented least squares algorithm which is implemented with dynamic programming
	 * and a successive method, which puts all values in a segment till the target loss is exceeded
	 */

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

	public static SegmentedRegression compressSegmentedLeastSquares(double[] column, CompressionSettings cs) {
		//compute Breakpoints for a Column with dynamic Programming
		final List<Integer> breakpointsList = computeBreakpoints(cs, column);
		final int[] breakpoints = breakpointsList.stream().mapToInt(Integer::intValue).toArray();

		//get values for Regression
		final int numSeg = breakpoints.length - 1;
		final double[] slopes = new double[numSeg];
		final double[] intercepts = new double[numSeg];

		// Regress per Segment
		for(int seg = 0; seg < numSeg; seg++) {
			final int SegStart = breakpoints[seg];
			final int SegEnd = breakpoints[seg + 1];

			final double[] line = regressSegment(column, SegStart, SegEnd);
			slopes[seg] = line[0]; //slope regession line
			intercepts[seg] = line[1]; //intercept regression line
		}

		return new SegmentedRegression(breakpoints, slopes, intercepts);
	}

	public static SegmentedRegression compressSuccessivePiecewiseLinear(double[] column, CompressionSettings cs) {
		//compute Breakpoints for a Column with a sukzessive breakpoints algorithm

		final List<Integer> breakpointsList = computeBreakpointSuccessive(column, cs);
		final int[] breakpoints = breakpointsList.stream().mapToInt(Integer::intValue).toArray();

		//get values for Regression
		final int numSeg = breakpoints.length - 1;
		final double[] slopes = new double[numSeg];
		final double[] intercepts = new double[numSeg];

		// Regress per Segment
		for(int seg = 0; seg < numSeg; seg++) {
			final int segstart = breakpoints[seg];
			final int segEnd = breakpoints[seg + 1];
			final double[] line = regressSegment(column, segstart, segEnd);
			slopes[seg] = line[0];
			intercepts[seg] = line[1];
		}
		return new SegmentedRegression(breakpoints, slopes, intercepts);
	}

	/**
	 * Computes breakpoints for a column using segmented least squares with dynamic programming
	 * Iteratively reduces lambda to increase the number of segments until the target MSE is met.
	 *
	 * @param cs     compression settings containing the target loss
	 * @param column the column values to segment
	 * @return list of breakpoint indices, starting with 0
	 */
	public static List<Integer> computeBreakpoints(CompressionSettings cs, double[] column) {
		final int numElements = column.length;
		final double targetMSE = cs.getPiecewiseTargetLoss();
		final double sseMax = numElements * targetMSE; // max allowed total SSE

		//start with high lambda an reduce iteratively
		double lambda = Math.max(10.0, sseMax * 2.0);
		List<Integer> bestBreaks = Arrays.asList(0, numElements);
		double bestSSE = computeTotalSSE(column, bestBreaks);

		for (int iter = 0; iter < 50; iter++) {
			List<Integer> breaks = computeBreakpointsLambda(column, lambda);
			double totalSSE = computeTotalSSE(column, breaks);
			int numSegs = breaks.size() - 1;

			if (totalSSE < bestSSE) {
				bestSSE = totalSSE;
				bestBreaks = new ArrayList<>(breaks);
			}
			//target loss reached
			if (bestSSE <= sseMax) {
				return bestBreaks;
			}

			// only one segment left, break condition
			if (numSegs <= 1) {
				break;
			}
			// reducing lambda to allow more segments in next iteration
			lambda *= 0.8;
		}

		return bestBreaks;
	}

	/**
	 * Computes optimal breakpoints, each segment has a SEE plus a

	 */

	public static List<Integer> computeBreakpointsLambda(double[] column, double lambda) {
		final int n = column.length;
		final double[] costs = new double[n + 1];  // min cost to reach i
		final int[] prev = new int[n + 1];

		Arrays.fill(costs, Double.POSITIVE_INFINITY);
		costs[0] = 0.0;
		// precompute all segment costs to avoid recomputation in dynamic programming
		double[][] segCosts = new double[n+1][n+1];
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j <= n; j++) {
				segCosts[i][j] = computeSegmentCost(column, i, j);
			}
		}
		// for each point j, find the cheapest previous breakpoint i
		for(int j = 1; j <= n; j++) {
			for(int i = 0; i < j; i++) {
				// cost equals the SSE of segment [i,j] plus penalty plus best costs
				double cost = costs[i] + segCosts[i][j] + lambda;
				if(cost < costs[j]) {
					costs[j] = cost;
					prev[j] = i;
				}
			}
		}

		// Backtrack to previous points to recover the breakpoints
		List<Integer> breaks = new ArrayList<>();
		int j = n;
		while(j > 0) {
			breaks.add(j);
			j = prev[j];
		}
		breaks.add(0);
		Collections.reverse(breaks);
		return breaks;
	}

	/**
	 * computes the segment cost
	 * @param column column values
	 * @param start start index
	 * @param end end index
	 * @return SSE of the regression line over the segment
	 */
	public static double computeSegmentCost(double[] column, int start, int end) {
		final int segSize = end - start;
		if(segSize <= 1)
			return 0.0;

		final double[] ab = regressSegment(column, start, end);
		final double slope = ab[0];
		final double intercept = ab[1];

		double sse = 0.0;
		for(int i = start; i < end; i++) {
			double err = column[i] - (slope * i + intercept);
			sse += err * err;
		}
		return sse;
	}

	/**
	 * computes the total SSE over all segments defined by the given breakpoints
	 * @param column
	 * @param breaks
	 * @return sum of the total SSE
	 */
	public static double computeTotalSSE(double[] column, List<Integer> breaks) {
		double total = 0.0;
		for(int s = 0; s < breaks.size() - 1; s++) {
			final int start = breaks.get(s);
			final int end = breaks.get(s + 1);
			total += computeSegmentCost(column, start, end);
		}
		return total;
	}

	public static double[] regressSegment(double[] column, int start, int end) {
		final int numElements = end - start;
		if(numElements <= 0)
			return new double[] {0.0, 0.0};

		double sumOfRowIndices = 0, sumOfColumnValues = 0, sumOfRowIndicesSquared = 0, productRowIndexTimesColumnValue = 0;
		for(int i = start; i < end; i++) {
			sumOfRowIndices += i;
			sumOfColumnValues += column[i];
			sumOfRowIndicesSquared += i * i;
			productRowIndexTimesColumnValue += i * column[i];
		}


		final double denominatorForSlope =
			numElements * sumOfRowIndicesSquared - sumOfRowIndices * sumOfRowIndices;
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
	 * computes breakpoints for a column using a successive algorithm
	 * extends each segment until the SEE reaches the target loss, then start a new segment
	 * @param column column values
	 * @param cs compression setting for setting the target loss
	 * @return list of breakpoint indices
	 */
	public static List<Integer> computeBreakpointSuccessive(double[] column, CompressionSettings cs) {
		final int numElements = column.length;
		final double targetMSE = cs.getPiecewiseTargetLoss();
		if (Double.isNaN(targetMSE) || targetMSE <= 0) {
			return Arrays.asList(0, numElements);  // fallback single segment
		}

		List<Integer> breakpoints = new ArrayList<>();
		breakpoints.add(0);
		int currentStart = 0;

		while (currentStart < numElements) {
			int bestEnd = -1; // no end found

			for (int end = currentStart + 1; end <= numElements; end++) {
				double sse = computeSegmentCost(column, currentStart, end);
				if(sse > (end - currentStart) * targetMSE) {
					// end-1 is last valid end; if end == segStart+1 force min segment of length 1
					bestEnd = (end == currentStart + 1) ? end : end - 1;
					break;
				}
			}

			if (bestEnd == -1) {
				bestEnd = numElements;// all remaining points fitting within budget
			}

			// safety guard not allow zero segments
			if (bestEnd <= currentStart) {
				bestEnd = Math.min(currentStart + 1, numElements);
			}

			breakpoints.add(bestEnd);
			currentStart = bestEnd;
		}

		// make sure, that the last breakpoint equals numElements
		int last = breakpoints.get(breakpoints.size() - 1);
		if (last != numElements) {
			breakpoints.add(numElements);
		}

		return breakpoints;
	}
}
