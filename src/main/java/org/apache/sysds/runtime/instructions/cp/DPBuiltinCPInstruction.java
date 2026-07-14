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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.dp.DPBudgetAccountant;

import java.util.LinkedHashMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * CP instruction for differential-privacy release of a linear query over the
 * original matrix.
 *
 * DML syntax (raw-matrix form):
 *   result = dp_laplace(X, query="colMeans", sensitivity=1.0, epsilon=0.5)
 *   result = dp_gaussian(X, query="colMeans", sensitivity=1.0, epsilon=0.5, delta=1e-5)
 *
 * The instruction receives the original {@code n x d} matrix {@code X},
 * builds a transformation matrix {@code T} ({@code k x n}) from the named
 * {@code query} (see {@link #buildTransform}), and returns a noisy release of
 * {@code T %*% X}. The noise is not added as a separate elementwise
 * pass over a materialised aggregate: it is injected by augmenting {@code T}
 * with an identity block and {@code X} with the noise matrix, so that the
 * noisy release is the result of a single {@link LibMatrixMult#matrixMult}
 * call (see {@link #processInstruction} for the derivation).
 *
 * Sensitivity norm: {@code sensitivity} is not interchangeable
 * between the two builtins. {@code dp_laplace} calibrates its noise scale
 * to the L1 sensitivity of {@code T %*% X} to a single-record
 * change; {@code dp_gaussian} calibrates its σ to the L2 sensitivity.
 * For a scalar release (e.g. {@code query="colMeans"} on single-column
 * {@code X}) the two norms coincide, but for a vector- or matrix-valued
 * release they generally differ — the caller is responsible for supplying
 * the norm matching the builtin invoked (see {@link #sensitivityOf}).
 *
 * The {@link #sensitivityOf} method is deliberately separated from the
 * noise-scale computation. It currently returns the caller-supplied
 * constant. A future rewrite pass could replace the body of this single
 * method with a static analysis that derives sensitivity from {@code T}'s
 * column norms and a declared per-record bound on {@code X}; every other
 * line in this class would stay unchanged.
 */
public class DPBuiltinCPInstruction extends ComputationCPInstruction {

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /** Opcode registered in Builtins and CPInstructionParser. */
    public static final String OPCODE_LAPLACE  = "dp_laplace";
    public static final String OPCODE_GAUSSIAN = "dp_gaussian";

    // -----------------------------------------------------------------------
    // Fields
    // -----------------------------------------------------------------------

    /**
     * Named parameters extracted from the serialised instruction string.
     * Keys: "target", "query", "sensitivity", "epsilon", "delta" (Gaussian only).
     *
     * Using the same LinkedHashMap<String,String> convention as
     * ParameterizedBuiltinCPInstruction so that CPInstructionParser can
     * call the shared constructParameterMap() helper unchanged.
     */
    private final LinkedHashMap<String, String> _params;

    // -----------------------------------------------------------------------
    // Constructor (private – use parseInstruction)
    // -----------------------------------------------------------------------

    private DPBuiltinCPInstruction(
            CPOperand input,
            CPOperand output,
            String opcode,
            String istr,
            LinkedHashMap<String, String> params) {
        super(CPType.DPBuiltin, null, input, null, output, opcode, istr);
        _params = params;
    }

    // -----------------------------------------------------------------------
    // Static factory / parser
    // -----------------------------------------------------------------------

    /**
     * Reconstructs a {@code DPBuiltinCPInstruction} from its serialised
     * instruction string produced by the LOP layer.
     *
     * Expected format (OPERAND_DELIM = '\u00b0'):
     *   dp_gaussian°target=mVar1·MATRIX·FP64°query=colMeans·SCALAR·STRING·true
     *              °sensitivity=1.0·SCALAR·FP64·true°epsilon=0.5·SCALAR·FP64·true
     *              °delta=1e-5·SCALAR·FP64·true°_mVar2·MATRIX·FP64
     *
     * The first token is always the opcode; the last token is always the
     * output operand; the tokens in between are key=value pairs. This matches
     * the convention used by ParameterizedBuiltinCPInstruction exactly.
     */
    public static DPBuiltinCPInstruction parseInstruction(String str) {
        String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
        InstructionUtils.checkNumFields(parts, 5, 6); // laplace=5, gaussian=6
        String opcode = parts[0];

        // Output operand is always the last token.
        CPOperand output = new CPOperand(parts[parts.length - 1]);

        // The "target" parameter holds the variable name of the input matrix.
        // ParameterizedBuiltinCPInstruction.constructParameterMap strips the
        // type suffixes and returns bare key=value pairs.
        LinkedHashMap<String, String> params =
                ParameterizedBuiltinCPInstruction.constructParameterMap(parts);

        // The target CPOperand is needed by ComputationCPInstruction's
        // getInputs() / getLineageItem() machinery.
        CPOperand input = new CPOperand(params.get("target"),
                org.apache.sysds.common.Types.ValueType.FP64,
                org.apache.sysds.common.Types.DataType.MATRIX);

        // Validate required keys.
        if (!params.containsKey("query"))
            throw new DMLRuntimeException(opcode + ": missing 'query'");
        if (!params.containsKey("sensitivity"))
            throw new DMLRuntimeException(opcode + ": missing 'sensitivity'");
        if (!params.containsKey("epsilon"))
            throw new DMLRuntimeException(opcode + ": missing 'epsilon'");
        if (opcode.equals(OPCODE_GAUSSIAN) && !params.containsKey("delta"))
            throw new DMLRuntimeException(opcode + ": missing 'delta'");

        return new DPBuiltinCPInstruction(input, output, opcode, str, params);
    }

    // -----------------------------------------------------------------------
    // Core execution
    // -----------------------------------------------------------------------

    /**
     * Executes the DP release.
     *
     *   - Read the original {@link MatrixBlock} {@code X} from the variable
     *       table.
     *   - Build the transformation matrix {@code T} ({@code k x n}) from
     *       {@code query} (see {@link #buildTransform}).
     *   - Determine sensitivity via {@link #sensitivityOf}.
     *   - Generate a noise {@link MatrixBlock} shaped {@code k x d}.
     *   - Fuse {@code T %*% X + noise} into a single
     *       {@link LibMatrixMult#matrixMult} call (see below).
     *   - Record the release with the session-scoped
     *       {@link DPBudgetAccountant}; throw if budget is exhausted.
     *   - Write the noisy block back to the variable table and release
     *       the input pin.
     *
     * Fusion derivation: for {@code T} ({@code k x n}), {@code X}
     * ({@code n x d}) and noise {@code N} ({@code k x d}), let
     * {@code T' = [T | I_k]} ({@code k x (n+k)}) and
     * {@code X' = [X ; N]} ({@code (n+k) x d}). Then
     * {@code T' %*% X' = T %*% X + I_k %*% N = T %*% X + N}, computed as one
     * matrix multiply instead of a multiply followed by a separate
     * elementwise add.
     */
    @Override
    public void processInstruction(ExecutionContext ec) {

        // ── 1. Read original input matrix X ─────────────────────────────────
        // getMatrixInput pins the block in memory and increments the
        // reference count; we must call releaseMatrixInput afterwards.
        MatrixBlock X = ec.getMatrixInput(_params.get("target"));

        // ── 2. Parse DP parameters ──────────────────────────────────────────
        double epsilon = parsePositiveDouble("epsilon");
        double delta   = instOpcode.equals(OPCODE_GAUSSIAN)
                         ? parsePositiveDouble("delta") : 0.0;
        String query = _params.get("query");

        // ── 3. Build the transformation matrix T (k x n) ────────────────────
        MatrixBlock T = buildTransform(query, X.getNumRows());

        // ── 4. Determine sensitivity (caller-supplied constant) ─────────────
        double sensitivity = sensitivityOf(T);

        // ── 5. Generate noise shaped like the release T %*% X (k x d) ───────
        MatrixBlock noiseBlock = generateNoise(T.getNumRows(), X.getNumColumns(),
                sensitivity, epsilon, delta);

        // ── 6. Fuse T %*% X + noise into a single matrix multiply ───────────
        MatrixBlock Ik = identity(T.getNumRows());
        MatrixBlock Tp = T.append(Ik, null, true);          // [T | I_k]
        MatrixBlock Xp = X.append(noiseBlock, null, false); // [X ; noise]
        MatrixBlock outBlock = LibMatrixMult.matrixMult(Tp, Xp);

        // ── 7. Record release and enforce budget ────────────────────────────
        // getDPBudgetAccountant() returns a lazy-initialised DPBudgetAccountant that is
        // owned by this ExecutionContext (added in a companion EC patch).
        DPBudgetAccountant accountant = ec.getDPBudgetAccountant();
        accountant.compose(epsilon, delta, sensitivity); // throws on exhaustion

        // ── 8. Write output and release input pin ───────────────────────────
        ec.releaseMatrixInput(_params.get("target"));
        ec.setMatrixOutput(output.getName(), outBlock);
    }

    // -----------------------------------------------------------------------
    // Transformation matrix construction
    // -----------------------------------------------------------------------

    /**
     * Builds the {@code k x n} transformation matrix {@code T} for the given
     * named query, to be left-multiplied against the {@code n x d} input
     * {@code X} as {@code T %*% X}.
     *
     *   - {@code "colMeans"}: {@code T} is {@code 1 x n}, filled with
     *       {@code 1/n} — {@code T %*% X} is the column-mean row vector.
     *   - {@code "colSums"}: {@code T} is {@code 1 x n}, filled with
     *       {@code 1.0} — {@code T %*% X} is the column-sum row vector.
     *   - {@code "identity"}: {@code T} is the {@code n x n} identity
     *       (built sparsely via {@link #identity}) — {@code T %*% X} is
     *       {@code X} itself, i.e. a noisy release of the raw matrix.
     *
     * Row-wise aggregates ({@code rowMeans}/{@code rowSums}) reduce across
     * the feature axis of {@code X}, i.e. they are naturally
     * {@code X %*% T'} (right-multiply), not {@code T %*% X}, so they are
     * intentionally not supported here.
     */
    private static MatrixBlock buildTransform(String query, int n) {
        switch (query) {
            case "colMeans": {
                MatrixBlock T = new MatrixBlock(1, n, false);
                T.allocateDenseBlock();
                double v = 1.0 / n;
                for (int c = 0; c < n; c++)
                    T.set(0, c, v);
                T.recomputeNonZeros();
                return T;
            }
            case "colSums": {
                MatrixBlock T = new MatrixBlock(1, n, false);
                T.allocateDenseBlock();
                for (int c = 0; c < n; c++)
                    T.set(0, c, 1.0);
                T.recomputeNonZeros();
                return T;
            }
            case "identity":
                return identity(n);
            default:
                throw new DMLRuntimeException(
                        "dp_laplace/dp_gaussian: unknown query type '" + query
                        + "' (expected colMeans, colSums, or identity)");
        }
    }

    /**
     * Builds a {@code k x k} identity matrix, sparsely, by reusing the
     * existing {@link LibMatrixReorg#diag} reorg operator (the same runtime
     * path DML's {@code diag()} builtin uses to expand a vector into a
     * diagonal matrix). Keeps memory {@code O(k)} rather than {@code O(k^2)},
     * which matters for the {@code query="identity"} case where {@code k}
     * equals the number of rows of {@code X}.
     */
    private static MatrixBlock identity(int k) {
        MatrixBlock ones = new MatrixBlock(k, 1, false);
        ones.allocateDenseBlock();
        for (int i = 0; i < k; i++)
            ones.set(i, 0, 1.0);
        ones.recomputeNonZeros();
        return LibMatrixReorg.diag(ones, new MatrixBlock(k, k, true));
    }

    // -----------------------------------------------------------------------
    // Sensitivity seam
    // -----------------------------------------------------------------------

    /**
     * Returns the sensitivity of the release {@code T %*% X} to a
     * single-record change, in the norm required by the mechanism actually
     * invoked: L1 for {@code dp_laplace}, L2 for
     * {@code dp_gaussian} (see the class Javadoc). The two only coincide
     * when the release is scalar.
     *
     * Returns the caller-supplied literal from the DML script as-is, with
     * no norm conversion or validation — the DML author must compute the
     * sensitivity in the correct norm for the builtin they call. A future
     * rewrite pass could replace this body with an analysis that derives
     * sensitivity from {@code T}'s column norms and a declared per-record
     * bound on {@code X}; no other line in this class would need to change.
     *
     * @param T the transformation matrix (unused for now; kept as the seam
     *          for a future sensitivity-derivation pass)
     * @return caller-supplied sensitivity constant, expected to already be
     *         in the L1 norm (Laplace) or L2 norm (Gaussian)
     */
    private double sensitivityOf(MatrixBlock T) {
        return parsePositiveDouble("sensitivity");
    }

    // -----------------------------------------------------------------------
    // Noise generation
    // -----------------------------------------------------------------------

    /**
     * Generates a {@code rows x cols} noise {@link MatrixBlock} — matching
     * the shape of the release {@code T %*% X} — filled with samples from the
     * mechanism-appropriate distribution calibrated to ({@code sensitivity},
     * {@code epsilon}, {@code delta}).
     *
     * Both mechanisms produce a dense block. Sparsity exploitation is
     * left for future work; for the releases targeted here (e.g. column
     * means, column sums) the noise is dense regardless.
     */
    private MatrixBlock generateNoise(
            int rows,
            int cols,
            double sensitivity,
            double epsilon,
            double delta) {

        MatrixBlock noise = new MatrixBlock(rows, cols, false); // dense
        noise.allocateDenseBlock();

        if (instOpcode.equals(OPCODE_LAPLACE)) {
            // Laplace mechanism
            // For a given epsilon, noise is drawn from the Laplace distribution at
            // scale b = sensitivity / epsilon
            fillLaplaceNoise(noise, sensitivity / epsilon);
        } else {
            // Gaussian mechanism: calibrate sigma for (epsilon, delta)-DP.
            // For a given epsilon, noise is drawn from the normal distribution at
            // sigma^2 = 2 * sensitivity^2 * log(1.25/delta) / epsilon^2
            double sigma = sensitivity
                    * Math.sqrt(2.0 * Math.log(1.25 / delta))
                    / epsilon;
            fillGaussianNoise(noise, sigma);
        }

        noise.recomputeNonZeros();
        return noise;
    }

    /**
     * Fills {@code block} with i.i.d. Laplace(0, scale) samples using the
     * inverse-CDF method.
     *
     * For u ~ Uniform(0, 1): X = -scale * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
     */
    private static void fillLaplaceNoise(MatrixBlock block, double scale) {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        int rows = block.getNumRows();
        int cols = block.getNumColumns();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double u = rng.nextDouble(); // u in (0, 1)
                double v = u - 0.5;
                // Guard against the degenerate u == 0.5 case (ln(0) = -inf).
                if (v == 0.0) v = 1e-15;
                double sample = -scale * Math.signum(v) * Math.log(1.0 - 2.0 * Math.abs(v));
                block.set(r, c, sample);
            }
        }
    }

    /**
     * Fills {@code block} with i.i.d. N(0, sigma²) samples.
     *
     * Uses {@link ThreadLocalRandom#nextGaussian()} which is thread-safe
     * and does not require external libraries.
     */
    private static void fillGaussianNoise(MatrixBlock block, double sigma) {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        int rows = block.getNumRows();
        int cols = block.getNumColumns();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                block.set(r, c, sigma * rng.nextGaussian());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /**
     * Parses a parameter value as a positive {@code double}.
     *
     * @throws DMLRuntimeException if the key is absent, unparseable, or
     *                             non-positive
     */
    private double parsePositiveDouble(String key) {
        String raw = _params.get(key);
        if (raw == null)
            throw new DMLRuntimeException(
                    instOpcode + ": parameter '" + key + "' is missing");
        double v;
        try {
            v = Double.parseDouble(raw);
        } catch (NumberFormatException e) {
            throw new DMLRuntimeException(
                    instOpcode + ": parameter '" + key
                    + "' is not a valid number: " + raw);
        }
        if (!(v > 0.0))
            throw new DMLRuntimeException(
                    instOpcode + ": parameter '" + key
                    + "' must be strictly positive, got " + v);
        return v;
    }
}
