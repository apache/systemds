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
import org.apache.sysds.runtime.instructions.cp.DPBuiltinOps;

/**
 * Session-scoped differential privacy budget accountant.
 *
 * Tracks composition of DP releases across the lifetime of a DML script execution. Each call to {@link #compose}
 * records one release and checks whether the cumulative privacy cost has exceeded the user-specified budget.
 *
 * The mechanism type (Laplace vs Gaussian) is inferred from the delta argument passed to {@link #compose}:
 *
 * - Laplace (delta == 0): pure epsilon-DP. The budget cost is tracked via basic composition: each release contributes
 * exactly its epsilon to a running sum. This is the tightest possible bound for pure DP and avoids the looser estimate
 * that results from routing Laplace through the RDP conversion path (which would introduce an unnecessary delta). Noise
 * scale is calibrated to L1 sensitivity (see {@link #compose}). - Gaussian (delta > 0): (epsilon, delta)-DP via Renyi
 * DP composition. Renyi divergences at a discrete set of orders alpha compose additively; the accumulated sum is
 * converted to (epsilon, delta) at query time using the formula from Mironov 2017. This is substantially tighter than
 * basic composition for repeated Gaussian releases, which is the common case in federated learning.
 *
 * When both mechanisms are used in the same script the total cost is:
 * epsilon_total = epsilon_Laplace_sum + epsilon_Gaussian_RDP
 * This follows from basic composition of a pure-DP mechanism with an approximate-DP mechanism, which is additive in epsilon.
 *
 * Renyi orders tracked (Gaussian path) alpha in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}. At query time the minimum
 * converted epsilon across all orders is taken as the tightest available bound.
 *
 * Gaussian RDP divergence For the Gaussian mechanism with noise scale sigma and L2 sensitivity:
 * D_alpha = alpha * sensitivity^2 / (2*sigma^2)
 * is back-derived from the caller's (epsilon, delta) via the standard calibration formula (see {@link #gaussianSigma}). Note that
 * sensitivity cancels in the final expression, so the RDP cost depends only on the (epsilon, delta) parameters.
 *
 * RDP => (epsilon, delta) conversion (Mironov 2017, Proposition 3):
 * epsilon(alpha) = R[alpha] + log(1/delta) / (alpha − 1)
 *
 * One instance is created per ExecutionContext (lazy init). It is garbage-collected with the context when the script
 * finishes; no state leaks between script executions or between concurrent scripts.
 *
 * Not thread-safe. A single DML script executes instructions sequentially on one thread, so no synchronisation is
 * needed.
 *
 * @see DPBuiltinOps
 */
public class DPBudgetAccountant {

	// -----------------------------------------------------------------------
	// Renyi orders used for Gaussian composition
	// -----------------------------------------------------------------------

	private static final double DEFAULT_EPSILON_BUDGET = 1.0;

	private static final double DEFAULT_DELTA = 1e-5;

	/**
	 * Discrete set of Renyi orders alpha. All must be > 1. Finer grids give tighter bounds; this set covers the range
	 * relevant for typical ML workloads.
	 */
	private static final double[] ORDERS = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

	// -----------------------------------------------------------------------
	// State
	// -----------------------------------------------------------------------

	/** Accumulated Renyi divergence at each order (Gaussian releases only). */
	private final double[] _rdpSum = new double[ORDERS.length];

	/**
	 * Running sum of pure epsilon from Laplace releases.
	 *
	 * Laplace gives pure epsilon-DP (no delta). Basic composition is exact and tighter than the RDP conversion path for
	 * Laplace (which would introduce an unnecessary delta and produce a looser bound). Each Laplace release adds its
	 * epsilon here; the total is added directly in {@link #totalEpsilonSpent()}.
	 */
	private double _pureEpsilonSum = 0.0;

	/** Total privacy budget (epsilon) for the script execution. */
	private final double _epsilonBudget;

	/** delta used for the Gaussian RDP-to-(epsilon,delta) conversion. */
	private final double _delta;

	/** Number of releases recorded so far (for error messages). */
	private int _releaseCount = 0;

	/** Whether at least one Gaussian release has been recorded. */
	private boolean _hasGaussianReleases = false;

	// -----------------------------------------------------------------------
	// Constructors
	// -----------------------------------------------------------------------

	/**
	 * Creates an accountant with the given global budget.
	 *
	 * Typical usage: the DML script sets the budget once at the top (the dp_set_budget(epsilon, delta) built-in), or
	 * the accountant is created with defaults and the budget is checked after each release.
	 *
	 * @param epsilonBudget total epsilon budget for the script execution (must be > 0)
	 * @param delta         delta used for the Gaussian RDP-to-(epsilon,delta) conversion (must be in (0,1))
	 */
	public DPBudgetAccountant(double epsilonBudget, double delta) {
		if(!(epsilonBudget > 0))
			throw new DMLRuntimeException("DPBudgetAccountant: epsilonBudget must be > 0, got " + epsilonBudget);
		if(!(delta > 0 && delta < 1))
			throw new DMLRuntimeException("DPBudgetAccountant: delta must be in (0,1), got " + delta);
		_epsilonBudget = epsilonBudget;
		_delta = delta;
	}

	/**
	 * Convenience constructor using a liberal default delta = 1e-5. Suitable when the calling script does not specify
	 * delta explicitly.
	 */
	public DPBudgetAccountant(double epsilonBudget) {
		this(epsilonBudget, 1e-5);
	}

	/**
	 * Default constructor using defaults. Suitable when the calling script does not specify epsilon, delta explicitly.
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
	 * This method must be called before the result is written to the variable table. If the budget is exhausted it
	 * throws and the caller's result is discarded, preventing an unaccounted release.
	 *
	 * Mechanism selection (see class-level Javadoc for details):
	 * - delta == 0 => Laplace, pure epsilon-DP basic composition
	 * - delta > 0 => Gaussian, Renyi DP composition
	 *
	 * @param epsilon     per-release epsilon parameter (must be >= 0)
	 * @param delta       per-release delta parameter (0 for Laplace, >= 0 for Gaussian)
	 * @param sensitivity sensitivity of the released quantity (must be > 0). The norm depends on the mechanism selected
	 *                    by delta: callers must supply the L1 sensitivity when delta == 0 (Laplace), and the L2
	 *                    sensitivity when delta > 0 (Gaussian). The two coincide for scalar-valued releases but diverge
	 *                    for vector-valued ones, so passing the wrong norm silently under- or over-calibrates the
	 *                    noise.
	 * @throws DMLRuntimeException if the cumulative epsilon after this release would exceed the budget
	 */
	public void compose(double epsilon, double delta, double sensitivity) {
		_releaseCount++;

		if(delta == 0.0) {
			// Laplace: pure epsilon-DP, basic composition - cost is exactly epsilon.
			_pureEpsilonSum += epsilon;
		}
		else {
			// Gaussian: accumulate Renyi divergence at each order, then convert.
			_hasGaussianReleases = true;
			for(int i = 0; i < ORDERS.length; i++) {
				double sigma = DPBuiltinOps.computeGaussianSigma(sensitivity, epsilon, delta);
				_rdpSum[i] += rdpGaussian(ORDERS[i], sensitivity, sigma);
			}
		}

		double spentEpsilon = totalEpsilonSpent();
		if(spentEpsilon > _epsilonBudget) {
			throw new DMLRuntimeException(String.format(
				"Privacy budget exhausted after %d release(s): "
					+ "spent epsilon %.6f exceeds budget epsilon = %.6f (delta = %.2e). "
					+ "Reduce the number of releases or widen the budget.",
				_releaseCount, spentEpsilon, _epsilonBudget, _delta));
		}
	}

	// -----------------------------------------------------------------------
	// Inspection
	// -----------------------------------------------------------------------

	/**
	 * Returns the current total privacy cost as an epsilon value.
	 *
	 * Total = Laplace pure-epsilon sum + Gaussian RDP-converted epsilon (clamped to zero when no Gaussian releases have
	 * been recorded).
	 */
	public double totalEpsilonSpent() {
		if(!_hasGaussianReleases)
			return _pureEpsilonSum;

		// Take min_alpha(epsilon_alpha) as the current total privacy cost
		double gaussianEps = Double.MAX_VALUE;
		for(int i = 0; i < ORDERS.length; i++) {
			double alpha = ORDERS[i];
			double eps = _rdpSum[i] + Math.log(1.0 / _delta) / (alpha - 1.0);
			if(eps < gaussianEps)
				gaussianEps = eps;
		}
		return _pureEpsilonSum + Math.max(gaussianEps, 0.0);
	}

	/** Returns the remaining epsilon budget (negative if the budget is exceeded). */
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
	 * Renyi divergence of order alpha for the Gaussian mechanism (Mironov 2017, Proposition 3, example 2):
	 * D_alpha = alpha * sensitivity^2 / (2 * sigma^2)
	 */
	private static double rdpGaussian(double alpha, double sensitivity, double sigma) {
		return alpha * (sensitivity * sensitivity) / (2.0 * sigma * sigma);
	}
}
