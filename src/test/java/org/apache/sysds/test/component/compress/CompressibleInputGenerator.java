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

package org.apache.sysds.test.component.compress;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * WARNING, this compressible input generator generates transposed inputs, (rows and cols are switched) this is because
 * then the test does not need to transpose the input for the colGroups that expect transposed inputs.
 * 
 */
public class CompressibleInputGenerator {
	protected static final Log LOG = LogFactory.getLog(CompressibleInputGenerator.class.getName());

	public static MatrixBlock getInput(int rows, int cols, CompressionType ct, int nrUnique, double sparsity, int seed) {
		return getInputDoubleMatrix(rows, cols, ct, nrUnique, 1000000, -1000000, sparsity, seed, false);
	}

	public static MatrixBlock getInput(int rows, int cols, CompressionType ct, int nrUnique, double sparsity, int seed,
		boolean transposed) {
		return getInputDoubleMatrix(rows, cols, ct, nrUnique, 1000000, -1000000, sparsity, seed, transposed);
	}

	public static MatrixBlock getInput(int rows, int cols, CompressionType ct, int nrUnique, int max, int min,
		double sparsity, int seed) {
		return getInputDoubleMatrix(rows, cols, ct, nrUnique, max, min, sparsity, seed, false);
	}

	public static MatrixBlock getInputDoubleMatrix(int rows, int cols, CompressionType ct, int nrUnique, int max,
		int min, double sparsity, int seed, boolean transpose) {
		final MatrixBlock output = transpose ? new MatrixBlock(cols, rows, sparsity < 0.4) : new MatrixBlock(rows, cols,
			sparsity < 0.4);
		if(nrUnique < 1)
			nrUnique = 1;
		switch(ct) {
			case RLE:
				rle(output, nrUnique, max, min, sparsity, seed, transpose);
				break;
			case OLE:
				ole(output, nrUnique, max, min, sparsity, seed, transpose);
				break;
			default:
				throw new NotImplementedException("Not implemented generator.");
		}
		return output;
	}

	public static MatrixBlock getInputOneHotMatrix(int rows, int cols, int seed) {
		MatrixBlock output = new MatrixBlock(rows, cols, true);
		Random r = new Random(seed);
		for(int i = 0; i < rows; i++)
			output.appendValue(i, r.nextInt(cols), 1);

		return output;
	}

	/**
	 * Generate a sparse matrix, that have less and less likelihood of values the more columns there is
	 * 
	 * @param rows
	 * @param cols
	 * @param unique
	 * @param max
	 * @param min
	 * @param seed
	 * @return
	 */
	public static MatrixBlock getUnbalancedSparseMatrix(int rows, int cols, int unique, int max, int min, int seed) {
		cols *= 3;
		MatrixBlock res = new MatrixBlock(rows, cols * 3, true);
		Random r = new Random(seed);
		final int range = max - min;
		final int multiplyer = unique != 0 ? range / unique : 1;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < Math.min(cols, 5); j++) {
				if(Math.sqrt(r.nextDouble() * 90) > (j + 0.3) * 2)
					if(unique == 0)
						res.appendValue(i, j, min);
					else
						res.appendValue(i, j, Math.round(Math.sqrt((double) r.nextInt(unique) * multiplyer + min)));
			}
		}
		res.examSparsity(true);
		return res;
	}

	private static void rle(MatrixBlock output, int nrUnique, int max, int min, double sparsity, int seed,
		boolean transpose) {

		final Random r = new Random(seed);
		final List<Double> values = getNRandomValues(nrUnique, r, max, min);

		final int cols = transpose ? output.getNumRows() : output.getNumColumns();
		final int rows = transpose ? output.getNumColumns() : output.getNumRows();

		for(int colNr = 0; colNr < cols; colNr++) {
			Collections.shuffle(values, r);

			// Generate a Dirichlet distribution, to distribute the values
			int[] occurences = makeDirichletDistribution(nrUnique, rows, r);

			int pointer = 0;
			int valuePointer = 0;
			for(int nr : occurences) {
				int zeros = (int) (Math.floor(nr * (1.0 - sparsity)));
				int before = (zeros > 0) ? r.nextInt(zeros) : 0;
				int after = zeros - before;
				pointer += before;
				for(int i = before; i < nr - after; i++) {
					if(transpose)
						output.quickSetValue(colNr, pointer, values.get(valuePointer));
					else
						output.quickSetValue(pointer, colNr, values.get(valuePointer));

					pointer++;
				}
				pointer += after;
				valuePointer++;
				if(valuePointer == values.size() && after == 0) {
					while(pointer < rows) {
						if(transpose)
							output.quickSetValue(colNr, pointer, values.get(nrUnique - 1));
						else
							output.quickSetValue(pointer, colNr, values.get(nrUnique - 1));
						pointer++;
					}
				}
			}
		}
	}

	private static void ole(MatrixBlock output, int nrUnique, int max, int min, double sparsity, int seed,
		boolean transpose) {

		// chose some random values
		final Random r = new Random(seed);
		final List<Double> values = getNRandomValues(nrUnique, r, max, min);
		if(transpose && output.isInSparseFormat() && output.getNumRows() == 1) {
			int nV = (int) Math.round((double) output.getNumColumns() * sparsity);
			int skip = (output.getNumColumns() * 2) / nV;

			for(int i = 0, n = 0; n < nV && i < output.getNumColumns(); i += r.nextInt(skip) + 1, n++)
				output.appendValue(0, i, values.get(r.nextInt(nrUnique)));

			output.getSparseBlock().sort();
			return;
		}

		final int cols = transpose ? output.getNumRows() : output.getNumColumns();
		final int rows = transpose ? output.getNumColumns() : output.getNumRows();

		// Generate the first column.
		for(int x = 0; x < rows; x++) {
			double d = values.get(r.nextInt(nrUnique));
			if(transpose && output.isInSparseFormat())
				output.appendValue(0, x, d);
			else if(transpose)
				output.quickSetValue(0, x, d);
			else
				output.quickSetValue(x, 0, d);
		}

		int diff = max - min;
		if(diff == 0)
			diff = 1;
		for(int y = 1; y < cols; y++) {
			for(int x = 0; x < rows; x++) {
				if(r.nextDouble() < sparsity) {
					if(transpose && output.isInSparseFormat()) {
						int v = (int) (output.getValue(0, x) * (double) y);
						double d = Math.abs(v % ((int) (diff))) + min;
						output.appendValue(y, x, d);
					}
					else if(transpose) {
						int v = (int) (output.getValue(0, x) * (double) y);
						double d = Math.abs(v % ((int) (diff))) + min;
						output.quickSetValue(y, x, d);
					}
					else {
						int v = (int) (output.getValue(x, 0) * (double) y);
						double d = Math.abs(v % ((int) (diff))) + min;
						output.quickSetValue(x, y, d);
					}
				}
			}
		}

		if(transpose && output.isInSparseFormat()) {
			SparseBlock sb = output.getSparseBlock();
			double[] r0 = sb.values(0);
			for(int i = 0; i < r0.length; i++)
				if(r.nextDouble() > sparsity)
					r0[i] = 0;
			sb.get(0).compact();
		}
		else {
			for(int x = 0; x < rows; x++) {
				if(r.nextDouble() > sparsity) {
					if(transpose)
						output.quickSetValue(0, x, 0);
					else
						output.quickSetValue(x, 0, 0);
				}
			}
		}

	}

	private static int[] makeDirichletDistribution(int nrUnique, int rows, Random r) {
		double[] distribution = new double[nrUnique];
		double sum = 0;
		for(int i = 0; i < nrUnique; i++) {
			distribution[i] = r.nextDouble();
			sum += distribution[i];
		}

		int[] occurences = new int[nrUnique];
		for(int i = 0; i < nrUnique; i++) {
			occurences[i] = (int) (((double) distribution[i] / (double) sum) * (double) rows);
		}
		return occurences;
	}

	private static List<Double> getNRandomValues(int nrUnique, Random r, int max, int min) {
		List<Double> values = new ArrayList<>();
		for(int i = 0; i < nrUnique; i++) {
			double v = Math.round(((r.nextDouble() * (max - min)) + min) * 100) / 100;
			values.add(Math.floor(v));
		}
		return values;
	}
}
