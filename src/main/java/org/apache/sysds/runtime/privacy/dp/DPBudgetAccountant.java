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
 * One instance is created per ExecutionContext. Each call to {@link #compose} records one release and checks whether
 * the cumulative privacy cost has exceeded the user-specified budget.
 *
 * The mechanism type (Laplace vs Gaussian) is inferred from the delta argument passed to {@link #compose}:
 * - Laplace (delta == 0): pure epsilon-DP.
 * - Gaussian (delta > 0): (epsilon, delta)-DP via Renyi DP composition.
 * When both mechanisms are used in the same script the total cost is:
 * epsilon_total = epsilon_Laplace_sum + epsilon_Gaussian_RDP
 * This follows from basic composition of a pure-DP mechanism with an approximate-DP mechanism, which is additive in epsilon.
 *
 * Not thread-safe. A single DML script executes instructions sequentially on one thread, so no synchronisation is
 * needed.
 */
public class DPBudgetAccountant {
	private static final double DEFAULT_EPSILON_BUDGET = 1.0;
	private static final double DEFAULT_DELTA = 1e-5;
	/**
	 * Discrete set of Renyi orders alpha. All must be > 1. Finer grids give tighter bounds; this set covers the range
	 * relevant for typical ML workloads.
	 */
	private static final double[] ORDERS = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
	/** Accumulated Renyi divergence at each order (Gaussian releases only). */
	private final double[] _rdpSum = new double[ORDERS.length];

	/**
	 * Running sum of pure epsilon from Laplace releases. Each Laplace release adds its
	 * epsilon here; the total is added directly in {@link #totalEpsilonSpent()}.
	 */
	private double _pureEpsilonSum = 0.0;
	/** Number of releases recorded so far (for error messages). */
	private int _releaseCount = 0;
	/** Whether at least one Gaussian release has been recorded. */
	private boolean _hasGaussianReleases = false;

	/** Total privacy budget (epsilon) for the script execution. */
	private final double _epsilonBudget;
	/** delta used for the Gaussian RDP-to-(epsilon,delta) conversion. */
	private final double _delta;

	/**
	 * Creates an accountant with the given global budget.
	 *
	 * @param epsilonBudget total epsilon budget for the script execution (must be > 0)
	 * @param delta         delta used for the Gaussian RDP-to-(epsilon,delta) conversion (must be in (0,1))
	 */
	public DPBudgetAccountant(double epsilonBudget, double delta) {
		if(epsilonBudget <= 0)
			throw new DMLRuntimeException("DPBudgetAccountant: epsilonBudget must be > 0, got " + epsilonBudget);
		if((delta <= 0) || (delta >= 1))
			throw new DMLRuntimeException("DPBudgetAccountant: delta must be in (0,1), got " + delta);
		_epsilonBudget = epsilonBudget;
		_delta = delta;
	}

	public DPBudgetAccountant(double epsilonBudget) {
		this(epsilonBudget, DEFAULT_DELTA);
	}

	public DPBudgetAccountant() {
		this(DEFAULT_EPSILON_BUDGET, DEFAULT_DELTA);
	}

	/**
	 * Records one DP release and checks the budget.
	 * This method must be called before the result is written to the variable table. If the budget is exhausted it
	 * throws and the caller's result is discarded, preventing an unaccounted release.
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
			// Laplace: pure epsilon-DP, basic composition; cost is exactly epsilon.
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

	/**
	 * Returns the current total privacy cost as an epsilon value.
	 */
	public double totalEpsilonSpent() {
		if(!_hasGaussianReleases)
			return _pureEpsilonSum;

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

	/**
	 * Renyi divergence of order alpha for the Gaussian mechanism (Mironov 2017, Proposition 3, example 2):
	 */
	private static double rdpGaussian(double alpha, double sensitivity, double sigma) {
		return alpha * (sensitivity * sensitivity) / (2.0 * sigma * sigma);
	}
}
