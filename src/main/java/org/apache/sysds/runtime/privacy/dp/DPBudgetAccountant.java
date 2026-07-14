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

package org.apache.sysds.runtime.privacy.dp;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.DPBuiltinCPInstruction;

/**
 * Session-scoped differential privacy budget accountant.
 *
 * <h2>Purpose</h2>
 * Tracks composition of DP releases across the lifetime of a DML script
 * execution. Each call to {@link #compose} records one release and checks
 * whether the cumulative privacy cost has exceeded the user-specified budget.
 *
 * <h2>Composition strategy</h2>
 * The mechanism type (Laplace vs Gaussian) is inferred from the {@code delta}
 * argument passed to {@link #compose}:
 *
 * <ul>
 *   <li><b>Laplace (delta == 0):</b> pure ε-DP. The budget cost is tracked via
 *       basic composition — each release contributes exactly its ε to a running
 *       sum. This is the tightest possible bound for pure DP and avoids the
 *       looser estimate that results from routing Laplace through the RDP
 *       conversion path (which would introduce an unnecessary δ). Noise scale
 *       is calibrated to <b>L1 sensitivity</b> (see {@link #compose}).</li>
 *   <li><b>Gaussian (delta &gt; 0):</b> (ε, δ)-DP via Rényi DP composition.
 *       Rényi divergences at a discrete set of orders α compose additively;
 *       the accumulated sum is converted to (ε, δ) at query time using the
 *       formula from Mironov 2017. This is substantially tighter than basic
 *       composition for repeated Gaussian releases, which is the common case
 *       in federated learning.</li>
 * </ul>
 *
 * <p>When both mechanisms are used in the same script the total cost is:
 * <pre>
 *   ε_total = ε_Laplace_sum + ε_Gaussian_RDP
 * </pre>
 * This follows from basic composition of a pure-DP mechanism with an
 * approximate-DP mechanism, which is additive in ε.
 *
 * <h2>Rényi orders tracked (Gaussian path)</h2>
 * α ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}. At query time the minimum
 * converted ε across all orders is taken as the tightest available bound.
 *
 * <h2>Gaussian RDP divergence</h2>
 * For the Gaussian mechanism with noise scale σ and L2 sensitivity Δf:
 * <pre>
 *   D_α = α · Δf² / (2σ²)
 * </pre>
 * σ is back-derived from the caller's (ε, δ) via the standard calibration
 * formula (see {@link #gaussianSigma}). Note that sensitivity cancels in the
 * final expression, so the RDP cost depends only on the (ε, δ) parameters.
 *
 * <h2>RDP → (ε, δ) conversion (Mironov 2017, Proposition 3)</h2>
 * <pre>
 *   ε(α) = R[α] + log(1 − 1/α) − log(δ·(α−1)) / α
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * One instance is created per {@code ExecutionContext} (lazy init). It is
 * garbage-collected with the context when the script finishes; no state
 * leaks between script executions or between concurrent scripts.
 *
 * <h2>Thread safety</h2>
 * Not thread-safe. A single DML script executes instructions sequentially
 * on one thread, so no synchronisation is needed.
 *
 * @see DPBuiltinCPInstruction
 */
public class DPBudgetAccountant {

    // -----------------------------------------------------------------------
    // Rényi orders used for Gaussian composition
    // -----------------------------------------------------------------------

    private static final double DEFAULT_EPSILON_BUDGET = 1.0;

    private static final double DEFAULT_DELTA = 1e-5;

    /**
     * Discrete set of Rényi orders α. All must be &gt; 1.
     * Finer grids give tighter bounds; this set covers the range relevant
     * for typical ML workloads.
     */
    private static final double[] ORDERS = {
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    };

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /** Accumulated Rényi divergence at each order (Gaussian releases only). */
    private final double[] _rdpSum = new double[ORDERS.length];

    /**
     * Running sum of pure ε from Laplace releases.
     *
     * <p>Laplace gives pure ε-DP (no δ). Basic composition is exact and
     * tighter than the RDP conversion path for Laplace (which would introduce
     * an unnecessary δ and produce a looser bound). Each Laplace release adds
     * its ε here; the total is added directly in {@link #totalEpsilonSpent()}.
     */
    private double _pureEpsilonSum = 0.0;

    /** Total privacy budget (ε) for the script execution. */
    private final double _epsilonBudget;

    /** δ used for the Gaussian RDP-to-(ε,δ) conversion. */
    private final double _delta;

    /** Number of releases recorded so far (for error messages). */
    private int _releaseCount = 0;

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /**
     * Creates an accountant with the given global budget.
     *
     * <p>Typical usage: the DML script sets the budget once at the top
     * (future work: a {@code dp_set_budget(epsilon, delta)} built-in),
     * or the accountant is created with defaults and the budget is checked
     * after each release.
     *
     * @param epsilonBudget total ε budget for the script execution (must be &gt; 0)
     * @param delta         δ used for the Gaussian RDP-to-(ε,δ) conversion (must be in (0,1))
     */
    public DPBudgetAccountant(double epsilonBudget, double delta) {
        if (!(epsilonBudget > 0))
            throw new DMLRuntimeException(
                    "DPBudgetAccountant: epsilonBudget must be > 0, got " + epsilonBudget);
        if (!(delta > 0 && delta < 1))
            throw new DMLRuntimeException(
                    "DPBudgetAccountant: delta must be in (0,1), got " + delta);
        _epsilonBudget = epsilonBudget;
        _delta = delta;
    }

    /**
     * Convenience constructor using a liberal default δ = 1e-5.
     * Suitable when the calling script does not specify δ explicitly.
     */
    public DPBudgetAccountant(double epsilonBudget) {
        this(epsilonBudget, 1e-5);
    }

    /**
     * Default constructor using defaults.
     * Suitable when the calling script does not specify ε, δ explicitly.
     */
    public DPBudgetAccountant() {
        this(DEFAULT_EPSILON_BUDGET, DEFAULT_DELTA);
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /**
     * Records one DP release and checks the budget.
     *
     * <p>This method must be called <em>before</em> the result is written to
     * the variable table. If the budget is exhausted it throws and the
     * caller's result is discarded, preventing an unaccounted release.
     *
     * <p>Mechanism selection (see class-level Javadoc for details):
     * <ul>
     *   <li>{@code delta == 0} → Laplace, pure ε-DP basic composition</li>
     *   <li>{@code delta > 0} → Gaussian, Rényi DP composition</li>
     * </ul>
     *
     * @param epsilon     per-release ε parameter (must be &gt; 0)
     * @param delta       per-release δ parameter (0 for Laplace, &gt;0 for Gaussian)
     * @param sensitivity sensitivity Δf of the released quantity (must be &gt; 0).
     *                    <b>The norm depends on the mechanism selected by
     *                    {@code delta}:</b> callers must supply the
     *                    <b>L1</b> sensitivity ‖f(D) − f(D′)‖₁ when
     *                    {@code delta == 0} (Laplace), and the <b>L2</b>
     *                    sensitivity ‖f(D) − f(D′)‖₂ when {@code delta > 0}
     *                    (Gaussian). The two coincide for scalar-valued
     *                    releases but diverge for vector-valued ones, so
     *                    passing the wrong norm silently under- or
     *                    over-calibrates the noise.
     * @throws DMLRuntimeException if the cumulative ε after this release
     *                             would exceed the budget
     */
    public void compose(double epsilon, double delta, double sensitivity) {
        _releaseCount++;

        if (delta == 0.0) {
            // Laplace: pure ε-DP, basic composition — cost is exactly epsilon.
            _pureEpsilonSum += epsilon;
        } else {
            // Gaussian: accumulate Rényi divergence at each order, then convert.
            for (int i = 0; i < ORDERS.length; i++) {
                double sigma = gaussianSigma(sensitivity, epsilon, delta);
                _rdpSum[i] += rdpGaussian(ORDERS[i], sensitivity, sigma);
            }
        }

        double spentEpsilon = totalEpsilonSpent();
        if (spentEpsilon > _epsilonBudget) {
            throw new DMLRuntimeException(String.format(
                    "Privacy budget exhausted after %d release(s): "
                    + "spent ε ≈ %.6f exceeds budget ε = %.6f (δ = %.2e). "
                    + "Reduce the number of releases or widen the budget.",
                    _releaseCount, spentEpsilon, _epsilonBudget, _delta));
        }
    }

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /**
     * Returns the current total privacy cost as an ε value.
     *
     * <p>Total = Laplace pure-ε sum + Gaussian RDP-converted ε (clamped to
     * zero when no Gaussian releases have been recorded).
     */
    public double totalEpsilonSpent() {
        // Take min_α(ε_α) as the current total privacy cost
        double gaussianEps = Double.MAX_VALUE;
        for (int i = 0; i < ORDERS.length; i++) {
            double alpha = ORDERS[i];
            double eps = _rdpSum[i]
                    + Math.log(1.0 - 1.0 / alpha)
                    - Math.log(_delta * (alpha - 1.0)) / alpha;
            if (eps < gaussianEps)
                gaussianEps = eps;
        }
        // Clamp: with no Gaussian releases the RDP sum is 0 and the log-delta
        // term alone drives gaussianEps to a small positive value; clamp to 0
        // so Laplace-only scripts are not penalised by δ they never requested.
        if (gaussianEps < 0) gaussianEps = 0.0;
        return _pureEpsilonSum + gaussianEps;
    }

    /** Returns the remaining ε budget (negative if the budget is exceeded). */
    public double remainingBudget() {
        return _epsilonBudget - totalEpsilonSpent();
    }

    /** Returns the number of DP releases recorded so far. */
    public int releaseCount() {
        return _releaseCount;
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /**
     * Rényi divergence of order α for the Gaussian mechanism (Mironov 2017,
     * Proposition 3, example 2):
     * <pre>
     *   D_α = α · Δf² / (2σ²)
     * </pre>
     */
    private static double rdpGaussian(double alpha, double sensitivity, double sigma) {
        return alpha * (sensitivity * sensitivity) / (2.0 * sigma * sigma);
    }

    /**
     * Gaussian noise scale σ calibrated to (ε, δ)-DP:
     * <pre>
     *   σ = Δf · sqrt(2 · log(1.25 / δ)) / ε
     * </pre>
     * Must match the formula used in {@link DPBuiltinCPInstruction} so that
     * the RDP cost recorded here is consistent with the noise actually injected.
     */
    private static double gaussianSigma(double sensitivity, double epsilon, double delta) {
        return sensitivity * Math.sqrt(2.0 * Math.log(1.25 / delta)) / epsilon;
    }
}
