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
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.privacy.dp.DPBudgetAccountant;

import java.util.LinkedHashMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * CP instruction for differential-privacy release of an already-computed
 * aggregate.
 *
 * <p>DML syntax (post-aggregate form):
 * <pre>
 *   result = dp_laplace(aggregate, sensitivity=1.0, epsilon=0.5)
 *   result = dp_gaussian(aggregate, sensitivity=1.0, epsilon=0.5, delta=1e-5)
 * </pre>
 *
 * <p>The instruction receives a materialised matrix (the aggregate result),
 * injects calibrated noise element-wise, records the release with the
 * session-scoped {@link DPBudgetAccountant}, and returns the noisy matrix.
 *
 * <p>Noise is generated in Java and added via a {@code MatrixBlock} binary
 * operation so that the output allocation path is identical to every other
 * CP instruction (no special memory-management required).
 *
 * <p>The {@link #sensitivityOf} method is deliberately separated from the
 * noise-scale computation. In Phase 1 it returns the caller-supplied
 * constant. In the future HOP-level rewrite pass (Phase 2) the body of this
 * single method is replaced with a static analysis that reads the
 * sensitivity bound computed by the compiler; every other line in this class
 * stays unchanged.
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
     * Keys: "target", "sensitivity", "epsilon", "delta" (Gaussian only).
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
     * <p>Expected format (OPERAND_DELIM = '\u00b0'):
     * <pre>
     *   dp_gaussian°target=mVar1·MATRIX·FP64°sensitivity=1.0·SCALAR·FP64·true
     *              °epsilon=0.5·SCALAR·FP64·true°delta=1e-5·SCALAR·FP64·true
     *              °_mVar2·MATRIX·FP64
     * </pre>
     *
     * The first token is always the opcode; the last token is always the
     * output operand; the tokens in between are key=value pairs. This matches
     * the convention used by ParameterizedBuiltinCPInstruction exactly.
     */
    public static DPBuiltinCPInstruction parseInstruction(String str) {
        String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
        InstructionUtils.checkNumFields(parts, 4, 5); // laplace=4, gaussian=5
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
     * <ol>
     *   <li>Read the aggregate {@link MatrixBlock} from the variable table.</li>
     *   <li>Determine sensitivity via {@link #sensitivityOf} (Phase-1 stub).</li>
     *   <li>Generate a noise {@link MatrixBlock} of the same shape.</li>
     *   <li>Add noise element-wise using the existing binary-operator path.</li>
     *   <li>Record the release with the session-scoped
     *       {@link DPBudgetAccountant}; throw if budget is exhausted.</li>
     *   <li>Write the noisy block back to the variable table and release
     *       the input pin.</li>
     * </ol>
     */
    @Override
    public void processInstruction(ExecutionContext ec) {

        // ── 1. Read aggregate input ─────────────────────────────────────────
        // getMatrixInput pins the block in memory and increments the
        // reference count; we must call releaseMatrixInput afterwards.
        MatrixBlock inBlock = ec.getMatrixInput(_params.get("target"));

        // ── 2. Parse DP parameters ──────────────────────────────────────────
        double epsilon = parsePositiveDouble("epsilon");
        double delta   = instOpcode.equals(OPCODE_GAUSSIAN)
                         ? parsePositiveDouble("delta") : 0.0;

        // ── 3. Determine sensitivity (Phase-1: caller-supplied constant) ────
        double sensitivity = sensitivityOf(inBlock);

        // ── 4. Generate and add noise ────────────────────────────────────────
        MatrixBlock noiseBlock = generateNoise(inBlock, sensitivity, epsilon, delta);

        // Element-wise addition via the standard binary-operator path.
        // binaryOperations allocates the output block internally.
        BinaryOperator plusOp = new BinaryOperator(Plus.getPlusFnObject());
        MatrixBlock outBlock = new MatrixBlock();
        inBlock.binaryOperations(plusOp, noiseBlock, outBlock);

        // ── 5. Record release and enforce budget ────────────────────────────
        // getDPBudgetAccountant() returns a lazy-initialised DPBudgetAccountant that is
        // owned by this ExecutionContext (added in a companion EC patch).
        DPBudgetAccountant accountant = ec.getDPBudgetAccountant();
        accountant.compose(epsilon, delta, sensitivity); // throws on exhaustion

        // ── 6. Write output and release input pin ───────────────────────────
        ec.releaseMatrixInput(_params.get("target"));
        ec.setMatrixOutput(output.getName(), outBlock);
    }

    // -----------------------------------------------------------------------
    // Sensitivity seam (Phase-1 stub; Phase-2 replaces this body only)
    // -----------------------------------------------------------------------

    /**
     * Returns the sensitivity of {@code aggregate} to a single-record change.
     *
     * <p><b>Phase 1 (now):</b> returns the caller-supplied literal from the
     * DML script. Sensitivity analysis is the caller's responsibility.
     *
     * <p><b>Phase 2 (HOP-level rewrite pass):</b> replace this body with a
     * call that inspects the HOP node that produced {@code aggregate}, reads
     * the {@code sensitivityBound} field computed during compilation, and
     * returns it. No other line in this class changes.
     *
     * @param aggregate the already-computed aggregate block (ignored in
     *                  Phase 1; used in Phase 2 to look up lineage)
     * @return caller-supplied sensitivity constant
     */
    private double sensitivityOf(MatrixBlock aggregate) {
        // Phase 1: unwrap the literal or variable value from the param map.
        // In Phase 2, replace the body below with HOP-annotation lookup.
        return parsePositiveDouble("sensitivity");
    }

    // -----------------------------------------------------------------------
    // Noise generation
    // -----------------------------------------------------------------------

    /**
     * Generates a noise {@link MatrixBlock} of the same shape as
     * {@code aggregate}, filled with samples from the mechanism-appropriate
     * distribution calibrated to ({@code sensitivity}, {@code epsilon},
     * {@code delta}).
     *
     * <p>Both mechanisms produce a dense block. Sparsity exploitation is
     * left for future work; for the aggregate outputs targeted here (e.g.
     * column means, row sums) the aggregate is already dense.
     */
    private MatrixBlock generateNoise(
            MatrixBlock aggregate,
            double sensitivity,
            double epsilon,
            double delta) {

        int rows = aggregate.getNumRows();
        int cols = aggregate.getNumColumns();
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
     * <p>For u ~ Uniform(0, 1): X = -scale * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
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
     * <p>Uses {@link ThreadLocalRandom#nextGaussian()} which is thread-safe
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
