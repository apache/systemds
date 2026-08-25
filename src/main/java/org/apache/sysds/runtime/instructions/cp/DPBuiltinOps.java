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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.Well1024a;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.privacy.dp.DPBudgetAccountant;

import java.util.Map;

/**
 * Differential-privacy release of a linear query over the original matrix, invoked from
 * {@link ParameterizedBuiltinCPInstruction} for the dp_laplace/dp_gaussian opcodes.
 *
 * DML syntax (raw-matrix form):
 * result = dp_laplace(X, query="colMeans", sensitivity=1.0, epsilon=0.5)
 * result = dp_gaussian(X, query="colMeans", sensitivity=1.0, epsilon=0.5, delta=1e-5)
 *
 * {@link #release} receives the original n x d matrix X, builds a transformation matrix T (k x n) from the named query
 * (see {@link #buildTransform}), and returns a noisy release of T %*% X. The noise is not added as a separate
 * elementwise pass over a materialised aggregate: it is injected by augmenting T with an identity block and X with the
 * noise matrix, so that the noisy release is the result of a single {@link LibMatrixMult#matrixMult} call (see
 * {@link #release} for the derivation).
 *
 * Sensitivity norm: sensitivity is not interchangeable between the two builtins. dp_laplace calibrates its noise scale
 * to the L1 sensitivity of T %*% X to a single-record change; dp_gaussian calibrates its stdev to the L2 sensitivity.
 * For a scalar release (e.g. query="colMeans" on single-column X) the two norms coincide, but for a vector- or
 * matrix-valued release they generally differ - the caller is responsible for supplying the norm matching the builtin
 * invoked (see {@link #sensitivityOf}).
 *
 * The {@link #sensitivityOf} method is deliberately separated from the noise-scale computation. It currently returns
 * the caller-supplied constant. A future rewrite pass could replace the body of this single method with a static
 * analysis that derives sensitivity from T's column norms and a declared per-record bound on X; every other line in
 * this class would stay unchanged.
 */
public class DPBuiltinOps {

	private static final NormalDistribution normal = new NormalDistribution();

	private DPBuiltinOps() {
		// static utility class
	}

	// -----------------------------------------------------------------------
	// Core execution
	// -----------------------------------------------------------------------

	/**
	 * Executes the DP release.
	 *
	 * - Build the transformation matrix T (k x n) from query (see {@link #buildTransform}).
	 * - Determine sensitivity via {@link #sensitivityOf}.
	 * - Generate a noise {@link MatrixBlock} shaped k x d.
	 * - Fuse T %*% X + noise into a single {@link LibMatrixMult#matrixMult} call (see below).
	 * - Record the release with the session-scoped {@link DPBudgetAccountant}; throw if budget is exhausted.
	 *
	 * Fusion derivation: for T (k x n), X (n x d) and noise N (k x d),
	 * let T' = [T | I_k] (k x (n+k)) and X' = [X ; N] ((n+k) x d). Then
	 * T' %*% X' = T %*% X + I_k %*% N = T %*% X + N, computed as one matrix multiply instead of a multiply
	 * followed by a separate elementwise add.
	 *
	 * @param X          the original input matrix (caller pins/releases it around this call)
	 * @param opcode     dp_laplace or dp_gaussian
	 * @param params     named parameters: "query", "sensitivity", "epsilon", "delta" (Gaussian only)
	 * @param accountant the session-scoped privacy budget accountant to charge this release against
	 * @return the noisy release T %*% X + N
	 */
	static MatrixBlock release(MatrixBlock X, String opcode, Map<String, String> params,
		DPBudgetAccountant accountant) {

		// ── 1. Parse DP parameters ──────────────────────────────────────────
		double epsilon = parsePositiveDouble(opcode, params, "epsilon");
		double delta = opcode.equalsIgnoreCase(Opcodes.DP_GAUSSIAN.toString()) ?
			parsePositiveDouble(opcode, params, "delta") : 0.0;
		String query = params.get("query");

		// ── 2. Build the transformation matrix T (k x n) ────────────────────
		MatrixBlock T = buildTransform(query, X.getNumRows());

		// ── 3. Determine sensitivity (caller-supplied constant) ─────────────
		double sensitivity = sensitivityOf(opcode, params, T);

		// ── 4. Generate noise shaped like the release T %*% X (k x d) ───────
		MatrixBlock noiseBlock = generateNoise(opcode, T.getNumRows(), X.getNumColumns(), sensitivity, epsilon, delta);

		// ── 5. Fuse T %*% X + noise into a single matrix multiply ───────────
		MatrixBlock Ik = identity(T.getNumRows());
		MatrixBlock Tp = T.append(Ik, null, true); // [T | I_k]
		MatrixBlock Xp = X.append(noiseBlock, null, false); // [X ; noise]
		MatrixBlock outBlock = LibMatrixMult.matrixMult(Tp, Xp);

		// ── 6. Record release and enforce budget ────────────────────────────
		accountant.compose(epsilon, delta, sensitivity); // throws on exhaustion

		return outBlock;
	}

	// -----------------------------------------------------------------------
	// Transformation matrix construction
	// -----------------------------------------------------------------------

	/**
	 * Builds the k x n transformation matrix T for the given named query, to be left-multiplied against the n x d input
	 * X as T %*% X.
	 *
	 * - "colMeans": T is 1 x n, filled with 1/n - T %*% X is the column-mean row vector.
	 * - "colSums": T is 1 x n, filled with 1.0 - T %*% X is the column-sum row vector.
	 * - "identity": T is the n x n identity (built sparsely via {@link #identity}) - T %*% X is X itself,
	 * i.e. a noisy release of the raw matrix.
	 *
	 * Row-wise aggregates (rowMeans/rowSums) reduce across the feature axis of X, i.e. they are naturally X %*% T'
	 * (right-multiply), not T %*% X, so they are intentionally not supported here.
	 */
	private static MatrixBlock buildTransform(String query, int n) {
		switch(query) {
			case "colMeans": {
				MatrixBlock T = new MatrixBlock(1, n, false);
				T.allocateDenseBlock();
				double v = 1.0 / n;
				for(int c = 0; c < n; c++)
					T.set(0, c, v);
				T.recomputeNonZeros();
				return T;
			}
			case "colSums": {
				MatrixBlock T = new MatrixBlock(1, n, false);
				T.allocateDenseBlock();
				for(int c = 0; c < n; c++)
					T.set(0, c, 1.0);
				T.recomputeNonZeros();
				return T;
			}
			case "identity":
				return identity(n);
			default:
				throw new DMLRuntimeException("dp_laplace/dp_gaussian: unknown query type '" + query
					+ "' (expected colMeans, colSums, or identity)");
		}
	}

	/**
	 * Builds a k x k identity matrix, sparsely, by reusing the existing {@link LibMatrixReorg#diag} reorg operator (the
	 * same runtime path DML's diag() builtin uses to expand a vector into a diagonal matrix). Keeps memory O(k) rather
	 * than O(k^2), which matters for the query="identity" case where k equals the number of rows of X.
	 */
	private static MatrixBlock identity(int k) {
		MatrixBlock ones = new MatrixBlock(k, 1, false);
		ones.allocateDenseBlock();
		for(int i = 0; i < k; i++)
			ones.set(i, 0, 1.0);
		ones.recomputeNonZeros();
		return LibMatrixReorg.diag(ones, new MatrixBlock(k, k, true));
	}

	// -----------------------------------------------------------------------
	// Sensitivity seam
	// -----------------------------------------------------------------------

	/**
	 * Returns the sensitivity of the release T %*% X to a single-record change, in the norm required by the mechanism
	 * actually invoked: L1 for dp_laplace, L2 for dp_gaussian (see the class Javadoc). The two only coincide when the
	 * release is scalar.
	 *
	 * Returns the caller-supplied literal from the DML script as-is, with no norm conversion or validation - the DML
	 * author must compute the sensitivity in the correct norm for the builtin they call. A future rewrite pass could
	 * replace this body with an analysis that derives sensitivity from T's column norms and a declared per-record bound
	 * on X; no other line in this class would need to change.
	 *
	 * @param T the transformation matrix (unused for now; kept as the seam for a future sensitivity-derivation pass)
	 * @return caller-supplied sensitivity constant, expected to already be in the L1 norm (Laplace) or L2 norm
	 *         (Gaussian)
	 */
	private static double sensitivityOf(String opcode, Map<String, String> params, MatrixBlock T) {
		return parsePositiveDouble(opcode, params, "sensitivity");
	}

	// -----------------------------------------------------------------------
	// Noise generation
	// -----------------------------------------------------------------------

	/**
	 * Generates a rows x cols noise {@link MatrixBlock} - matching the shape of the release T %*% X - filled with
	 * samples from the mechanism-appropriate distribution calibrated to (sensitivity, epsilon, delta).
	 *
	 * Both mechanisms produce a dense block. Sparsity exploitation is left for future work; for the releases targeted
	 * here (e.g. column means, column sums) the noise is dense regardless.
	 */
	private static MatrixBlock generateNoise(String opcode, int rows, int cols, double sensitivity, double epsilon,
		double delta) {

		MatrixBlock noise;

		if(opcode.equalsIgnoreCase(Opcodes.DP_LAPLACE.toString())) {
			// Laplace mechanism
			// For a given epsilon, noise is drawn from the Laplace distribution at
			// scale b = sensitivity / epsilon
			noise = fillLaplaceNoise(rows, cols, sensitivity / epsilon);
		}
		else {
			// Gaussian mechanism
			// For a given epsilon and delta, noise is drawn from the Gaussian distribution
			// N(0, sigma^2)
			double sigma = computeGaussianSigma(sensitivity, epsilon, delta);
			noise = fillGaussianNoise(rows, cols, sigma);
		}

		return noise;
	}

	/**
	 * Compute the optimal sigma for the Analytic Gaussian Mechanism (Balle & Wang 2018). Returns the smallest sigma
	 * such that the Gaussian mechanism is (epsilon, delta)-DP.
	 *
	 * @param sensitivity L2 sensitivity
	 * @param epsilon     target epsilon
	 * @param delta       target delta
	 * @return optimal sigma
	 */
	public static double computeGaussianSigma(double sensitivity, double epsilon, double delta) {

		// Upper bound: classical Gaussian mechanism (loose but safe)
		double sigmaHigh = (sensitivity * Math.sqrt(2 * Math.log(1.25 / delta))) / epsilon;
		double sigmaLow = 1e-12;

		for(int i = 0; i < 100; i++) {
			double sigmaMid = 0.5 * (sigmaLow + sigmaHigh);

			if(deltaUpperBound(sigmaMid, epsilon, delta, sensitivity) > delta) {
				sigmaLow = sigmaMid;
			}
			else {
				sigmaHigh = sigmaMid;
			}
		}

		return sigmaHigh;
	}

	/**
	 * Analytic Gaussian DP inequality from Balle & Wang (2018).
	 */
	private static double deltaUpperBound(double sigma, double epsilon, double delta, double sensitivity) {
		double c = sensitivity / (2 * sigma);

		double term1 = normal.cumulativeProbability(c - epsilon * sigma / sensitivity);
		double term2 = Math.exp(epsilon) * normal.cumulativeProbability(-c - epsilon * sigma / sensitivity);

		return term1 - term2;
	}

	/**
	 * Fills block with i.i.d. Laplace(0, scale) samples.
	 *
	 * Draws from commons-math3's {@link LaplaceDistribution} (its inverse-CDF sampling, tested independently of this
	 * class) seeded by {@link Well1024a}, the same long-period equidistributed generator
	 * {@link org.apache.sysds.runtime.matrix.data.LibMatrixDatagen} uses for DML's rand() builtin.
	 */
	private static MatrixBlock fillLaplaceNoise(int rows, int cols, double scale) {
		MatrixBlock noise = new MatrixBlock(rows, cols, false); // dense
		noise.allocateDenseBlock();
		LaplaceDistribution laplace = new LaplaceDistribution(new Well1024a(), 0, scale);
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < cols; c++) {
				noise.set(r, c, laplace.sample());
			}
		}
		return noise;
	}

	/**
	 * Generates a rows x cols block of i.i.d. N(0, sigma^2) samples.
	 *
	 * Reuses the same Well1024a-seeded, Box-Muller normal generator that backs DML's rand(pdf="normal") (see
	 * {@link MatrixBlock#randOperations}), so the noise gets the same long-period PRNG and block-parallel generation as
	 * the rest of SystemDS's random matrix generation. randOperations produces standard N(0,1) samples (pdf="normal"
	 * ignores min/max), so the sigma scaling is applied afterwards as a scalar multiply.
	 */
	private static MatrixBlock fillGaussianNoise(int rows, int cols, double sigma) {
		MatrixBlock std = MatrixBlock.randOperations(rows, cols, 1.0, 0, 1, "normal", -1);
		return std.scalarOperations(new RightScalarOperator(Multiply.getMultiplyFnObject(), sigma), null);
	}

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	/**
	 * Parses a parameter value as a positive double.
	 *
	 * @throws DMLRuntimeException if the key is absent, unparseable, or non-positive
	 */
	private static double parsePositiveDouble(String opcode, Map<String, String> params, String key) {
		String raw = params.get(key);
		if(raw == null)
			throw new DMLRuntimeException(opcode + ": parameter '" + key + "' is missing");
		double v;
		try {
			v = Double.parseDouble(raw);
		}
		catch(NumberFormatException e) {
			throw new DMLRuntimeException(opcode + ": parameter '" + key + "' is not a valid number: " + raw);
		}
		if(!(v > 0.0))
			throw new DMLRuntimeException(opcode + ": parameter '" + key + "' must be strictly positive, got " + v);
		return v;
	}
}
