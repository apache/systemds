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

/**
 * Session-scoped Rényi Differential Privacy (RDP) budget accountant.
 *
 * <h2>Purpose</h2>
 * Tracks composition of DP releases across the lifetime of a DML script
 * execution. Each call to {@link #compose} records one release and checks
 * whether the cumulative privacy cost has exceeded the user-specified budget.
 *
 * <h2>Why Rényi DP?</h2>
 * Basic composition adds epsilons linearly, giving very loose bounds with
 * many releases. Rényi DP divergences compose <em>additively</em> at the
 * same order α. Converting the running Rényi sum to (ε, δ) via the standard
 * conversion formula yields substantially tighter bounds — particularly for
 * Gaussian mechanisms, which are common in federated learning.
 *
 * <h2>Orders tracked</h2>
 * We track a discrete set of Rényi orders α ∈ {2, 4, 8, 16, 32, 64, 128,
 * 256, 512, 1024}. At query time we take the minimum converted ε across all
 * orders, which is the tightest available bound.
 *
 * <h2>Composition rules</h2>
 * For the Gaussian mechanism with noise scale σ and sensitivity Δf, the
 * Rényi divergence of order α between outputs on neighbouring datasets is:
 * <pre>
 *   D_α = α · Δf² / (2σ²)
 * </pre>
 * where σ is back-derived from the caller's (ε, δ) parameters via the
 * standard calibration formula. See {@link #rdpGaussian} for details.
 *
 * For the Laplace mechanism with scale b = Δf/ε, the Rényi divergence at
 * order α is:
 * <pre>
 *   D_α = (1/(α-1)) · ln( α/(2α-1) · exp((α-1)/b) + (α-1)/(2α-1) · exp(-α/b) )
 *       (for α > 1; the limit as α → 1 is 1/b, i.e. the KL divergence)
 * </pre>
 *
 * <h2>Conversion: Rényi DP → (ε, δ)-DP</h2>
 * Given accumulated Rényi divergence R[α] at order α and a target δ:
 * <pre>
 *   ε(α) = R[α] + log(1 - 1/α) - log(δ · (α - 1)) / α
 * </pre>
 * The reported total cost is min_α ε(α).
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
public class RDPAccountant {

    // -----------------------------------------------------------------------
    // Rényi orders to track
    // -----------------------------------------------------------------------

    /**
     * Discrete set of Rényi orders α. All must be > 1.
     * Finer grids give tighter bounds; this set is a reasonable default
     * that covers the range relevant for typical ML workloads.
     */
    private static final double[] ORDERS = {
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    };

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /** Running accumulated Rényi divergence at each order. */
    private final double[] _rdpSum = new double[ORDERS.length];

    /** User-specified total privacy budget (ε). */
    private final double _epsilonBudget;

    /** User-specified δ used for the RDP-to-(ε,δ) conversion. */
    private final double _delta;

    /** Number of releases recorded so far (for error messages). */
    private int _releaseCount = 0;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates an accountant with the given global budget.
     *
     * <p>Typical usage: the DML script sets the budget once at the top
     * (future work: a {@code dp_set_budget(epsilon, delta)} built-in),
     * or the accountant is created with defaults and the budget is checked
     * after each release.
     *
     * @param epsilonBudget total ε budget for the script execution (must be > 0)
     * @param delta         δ used for the RDP-to-(ε,δ) conversion (must be in (0,1))
     */
    public RDPAccountant(double epsilonBudget, double delta) {
        if (!(epsilonBudget > 0))
            throw new DMLRuntimeException(
                    "RDPAccountant: epsilonBudget must be > 0, got " + epsilonBudget);
        if (!(delta > 0 && delta < 1))
            throw new DMLRuntimeException(
                    "RDPAccountant: delta must be in (0,1), got " + delta);
        _epsilonBudget = epsilonBudget;
        _delta = delta;
    }

    /**
     * Convenience constructor using a liberal default δ = 1e-5.
     * Suitable when the calling script does not specify δ explicitly.
     */
    public RDPAccountant(double epsilonBudget) {
        this(epsilonBudget, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    /**
     * Records one DP release and checks the budget.
     *
     * <p>This method must be called <em>before</em> the result is written to
     * the variable table. If the budget is exhausted, it throws and the
     * caller's result is discarded, preventing an unaccounted release.
     *
     * <p>The mechanism type (Laplace vs Gaussian) is inferred from the
     * parameters: if {@code delta == 0} the release is treated as Laplace;
     * otherwise it is treated as Gaussian.
     *
     * @param epsilon     the ε parameter for this individual release (> 0)
     * @param delta       the δ parameter for this release (0 for Laplace)
     * @param sensitivity the L2 sensitivity Δf of the released quantity (> 0)
     * @throws DMLRuntimeException if the cumulative ε after this release
     *                             would exceed the budget
     */
    public void compose(double epsilon, double delta, double sensitivity) {
        _releaseCount++;

        // Accumulate Rényi divergence at each order.
        for (int i = 0; i < ORDERS.length; i++) {
            double alpha = ORDERS[i];
            double rdp;
            if (delta == 0.0) {
                rdp = rdpLaplace(alpha, sensitivity, epsilon);
            } else {
                // Back-derive σ from the (ε, δ) calibration formula for the
                // Gaussian mechanism, then compute the RDP contribution.
                double sigma = gaussianSigma(sensitivity, epsilon, delta);
                rdp = rdpGaussian(alpha, sensitivity, sigma);
            }
            _rdpSum[i] += rdp;
        }

        // Convert accumulated RDP to (ε, δ) and check.
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
     * Returns the current total privacy cost as an ε value under the
     * accountant's δ, using the tightest available Rényi order.
     */
    public double totalEpsilonSpent() {
        double minEpsilon = Double.MAX_VALUE;
        for (int i = 0; i < ORDERS.length; i++) {
            double alpha = ORDERS[i];
            // Standard RDP-to-(ε,δ) conversion:
            //   ε(α) = R[α] + log(1 - 1/α) - log(δ·(α-1)) / α
            // Reference: Mironov 2017, Proposition 3.
            double eps = _rdpSum[i]
                    + Math.log(1.0 - 1.0 / alpha)
                    - Math.log(_delta * (alpha - 1.0)) / alpha;
            if (eps < minEpsilon)
                minEpsilon = eps;
        }
        return minEpsilon;
    }

    /** Returns the remaining ε budget (may be negative if budget is exceeded). */
    public double remainingBudget() {
        return _epsilonBudget - totalEpsilonSpent();
    }

    /** Returns the number of DP releases recorded so far. */
    public int releaseCount() {
        return _releaseCount;
    }

    // -----------------------------------------------------------------------
    // Mechanism-specific RDP contributions
    // -----------------------------------------------------------------------

    /**
     * Rényi divergence of order α for the Laplace mechanism with scale
     * b = sensitivity / epsilon.
     *
     * <p>For α > 1 and integer α, the closed form is:
     * <pre>
     *   D_α = (1/(α-1)) · ln( α/(2α-1)·exp((α-1)/b) + (α-1)/(2α-1)·exp(-α/b) )
     * </pre>
     *
     * <p>For non-integer α we use the same formula, which is the natural
     * analytic continuation (see Mironov 2017, Proposition 3, example 1).
     * We clamp the log argument to avoid NaN when inputs are degenerate.
     */
    private static double rdpLaplace(double alpha, double sensitivity, double epsilon) {
        double b = sensitivity / epsilon; // Laplace scale
        double t1 = alpha / (2.0 * alpha - 1.0) * Math.exp((alpha - 1.0) / b);
        double t2 = (alpha - 1.0) / (2.0 * alpha - 1.0) * Math.exp(-alpha / b);
        double arg = t1 + t2;
        if (arg <= 0) return 0.0; // degenerate: treat as zero cost
        return Math.log(arg) / (alpha - 1.0);
    }

    /**
     * Rényi divergence of order α for the Gaussian mechanism with noise
     * scale σ and L2 sensitivity Δf.
     *
     * <p>For α > 1:
     * <pre>
     *   D_α = α · Δf² / (2σ²)
     * </pre>
     *
     * <p>This is the standard result for the Gaussian mechanism (see
     * Mironov 2017, Proposition 3, example 2).
     */
    private static double rdpGaussian(double alpha, double sensitivity, double sigma) {
        return alpha * (sensitivity * sensitivity) / (2.0 * sigma * sigma);
    }

    /**
     * Back-derives the Gaussian noise scale σ from the (ε, δ)-DP parameters
     * using the standard calibration inequality:
     * <pre>
     *   σ = Δf · sqrt(2 · ln(1.25 / δ)) / ε
     * </pre>
     *
     * <p>This is the formula used by {@code DPBuiltinCPInstruction} to
     * generate the actual noise, so the RDP contribution it records is
     * exactly consistent with the noise injected.
     */
    private static double gaussianSigma(double sensitivity, double epsilon, double delta) {
        return sensitivity * Math.sqrt(2.0 * Math.log(1.25 / delta)) / epsilon;
    }
}
