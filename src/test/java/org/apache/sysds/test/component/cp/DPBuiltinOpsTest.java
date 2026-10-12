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

package org.apache.sysds.test.component.cp;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.DPBuiltinOps;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.dp.DPBudgetAccountant;

import java.lang.reflect.Method;

import org.junit.Test;
import org.junit.Assert;

/**
 * Tests for DPBuiltinOps, DPBuiltinCPInstruction (dp_laplace/dp_gaussian), and DPBudgetAccountant.
 */
public class DPBuiltinOpsTest {

	private static final double ASSERT_TOLERANCE = 1e-9;

	// =======================================================================
	// 1. DPBudgetAccountant unit tests
	// =======================================================================

	@Test
	public void testAccountantInitialisesAtZeroCostThenAcceptsSingleReleases() {
		DPBudgetAccountant acc = new DPBudgetAccountant(1.0, 1e-5);
		// No releases yet: remainingBudget() should be positive.
		Assert.assertTrue("No releases should leave budget intact", acc.remainingBudget() > 0);
		Assert.assertEquals(0, acc.releaseCount());

		// epsilon=0.5, budget=1.0: one Laplace release should stay within the budget.
		acc.compose(0.5, 0.0, 1.0); // Laplace
		Assert.assertEquals(1, acc.releaseCount());
		Assert.assertTrue("Single release within budget", acc.totalEpsilonSpent() <= 1.0);

		// Likewise, a single Gaussian release on a fresh accountant should stay within budget.
		DPBudgetAccountant gaussianAcc = new DPBudgetAccountant(1.0, 1e-5);
		gaussianAcc.compose(0.5, 1e-5, 1.0); // Gaussian
		Assert.assertEquals(1, gaussianAcc.releaseCount());
		Assert.assertTrue("Single Gaussian release within budget", gaussianAcc.totalEpsilonSpent() <= 1.0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testBudgetExhaustionThrows() {
		// Budget is set to only 0.5, but we try to make 10 releases with epsilon of 0.1 each.
		// After enough releases the budget must be exceeded.
		DPBudgetAccountant acc = new DPBudgetAccountant(0.5, 1e-5);
		double prevEpsilonSpent = acc.totalEpsilonSpent();
		double prevRemainingBudget = acc.remainingBudget();
		for(int i = 0; i < 10; i++) {
			acc.compose(0.1, 0.0, 1.0); // will throw before the 10th
			double currentEpsilonSpent = acc.totalEpsilonSpent();
			double currentRemainingBudget = acc.remainingBudget();
			int releaseCount = acc.releaseCount();
			Assert.assertTrue("Epsilon spent must increase with each release", currentEpsilonSpent > prevEpsilonSpent);
			Assert.assertTrue("Remaining budget must decrease", currentRemainingBudget < prevRemainingBudget);
			Assert.assertEquals("Release count must match", i + 1, releaseCount);
			prevEpsilonSpent = currentEpsilonSpent;
			prevRemainingBudget = currentRemainingBudget;
		}
	}

	@Test
	public void testGaussianTighterThanLaplaceForSameEpsilon() {
		// Gaussian uses RDP composition which is tighter than Laplace with basic composition.
		double eps = 0.5;
		double delta = 1e-5;

		DPBudgetAccountant gaussian = new DPBudgetAccountant(100.0, delta);
		DPBudgetAccountant laplace = new DPBudgetAccountant(100.0, delta);

		for(int i = 0; i < 5; i++) {
			gaussian.compose(eps, delta, 1.0);
			laplace.compose(eps, 0.0, 1.0);
		}

		Assert.assertTrue("Gaussian RDP bound should be tighter than Laplace bound after 5 releases",
			gaussian.totalEpsilonSpent() <= laplace.totalEpsilonSpent() + 1e-6);
	}

	@Test
	public void testHigherEpsilonCostMoreForLaplace() {
		// Increasing the epsilon results in higher budget costs.
		DPBudgetAccountant acc1 = new DPBudgetAccountant(100.0, 1e-5);
		DPBudgetAccountant acc2 = new DPBudgetAccountant(100.0, 1e-5);
		acc1.compose(0.5, 0.0, 1.0); // epsilon=0.5, Laplace
		acc2.compose(1.0, 0.0, 1.0); // epsilon=1.0, same sensitivity

		Assert.assertTrue("Higher epsilon costs more budget (Laplace basic composition)",
			acc1.totalEpsilonSpent() < acc2.totalEpsilonSpent());
	}

	// --- Constructor error paths ------------------------------------

	@Test
	public void testConstructorRejectsInvalidBudgetParameters() {
		assertConstructorRejects(0.0, 1e-5); // zero epsilon budget
		assertConstructorRejects(-0.5, 1e-5); // negative epsilon budget
		assertConstructorRejects(1.0, 0.0); // delta = 0
		assertConstructorRejects(1.0, 1.0); // delta = 1
	}

	private static void assertConstructorRejects(double epsilonBudget, double delta) {
		try {
			new DPBudgetAccountant(epsilonBudget, delta);
			Assert.fail("Expected DMLRuntimeException for epsilonBudget=" + epsilonBudget + ", delta=" + delta);
		}
		catch(DMLRuntimeException e) {
			// expected
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void testGaussianBudgetExhaustionThrows() {
		// Budget = 0.5. Each Gaussian release is composed at epsilon=0.3, delta=1e-5,
		// so the RDP-converted cost accumulates well past the budget within 20 releases.
		DPBudgetAccountant acc = new DPBudgetAccountant(0.5, 1e-5);
		for(int i = 0; i < 20; i++) {
			acc.compose(0.3, 1e-5, 1.0);
		}
	}

	@Test
	public void testMixedCompositionExceedsEitherAlone() {
		// Compose one Laplace and one Gaussian release. The total cost must
		// exceed what either mechanism contributes alone.
		DPBudgetAccountant mixed = new DPBudgetAccountant(100.0, 1e-5);
		DPBudgetAccountant lapOnly = new DPBudgetAccountant(100.0, 1e-5);
		DPBudgetAccountant gauOnly = new DPBudgetAccountant(100.0, 1e-5);

		mixed.compose(0.5, 0.0, 1.0); // Laplace
		mixed.compose(0.5, 1e-5, 1.0); // Gaussian

		lapOnly.compose(0.5, 0.0, 1.0);
		gauOnly.compose(0.5, 1e-5, 1.0);

		Assert.assertTrue("Mixed cost must exceed Laplace-only cost",
			mixed.totalEpsilonSpent() > lapOnly.totalEpsilonSpent());
		Assert.assertTrue("Mixed cost must exceed Gaussian-only cost",
			mixed.totalEpsilonSpent() > gauOnly.totalEpsilonSpent());
	}

	@Test
	public void testGaussianSensitivityCancelsInRDP() {
		// Two accountants with the same (epsilon,delta) but different sensitivity must
		// report identical totalEpsilonSpent().
		DPBudgetAccountant acc1 = new DPBudgetAccountant(100.0, 1e-5);
		DPBudgetAccountant acc2 = new DPBudgetAccountant(100.0, 1e-5);
		acc1.compose(0.5, 1e-5, 1.0);
		acc2.compose(0.5, 1e-5, 100.0);
		Assert.assertEquals("Gaussian RDP cost must be independent of sensitivity when (epsilon,delta) are fixed",
			acc1.totalEpsilonSpent(), acc2.totalEpsilonSpent(), ASSERT_TOLERANCE);
	}

	@Test
	public void testGaussianLargerEpsilonCostsMoreBudget() {
		// Increasing epsilon results in higher budget costs for Gaussian.
		DPBudgetAccountant lowEps = new DPBudgetAccountant(100.0, 1e-5);
		DPBudgetAccountant highEps = new DPBudgetAccountant(100.0, 1e-5);
		lowEps.compose(0.1, 1e-5, 1.0);
		highEps.compose(0.5, 1e-5, 1.0);
		Assert.assertTrue("Larger epsilon per Gaussian release must cost more budget",
			highEps.totalEpsilonSpent() > lowEps.totalEpsilonSpent());
	}

	// =======================================================================
	// 2. Noise distribution tests (statistical sanity checks)
	// =======================================================================
	// These tests generate many samples and verify that the empirical mean
	// is near zero and the empirical variance matches the theoretical value
	// within a reasonable tolerance.

	@Test
	public void testLaplaceNoiseDistribution() throws ReflectiveOperationException {
		// For 10000 samples the empirical mean should be within 5*sigma/sqrt(n) of 0,
		// and Var[Laplace(0, b)] = 2b^2 should match within 10% relative error.
		int n = 10_000;
		double scale = 1.5;
		double[] samples = sampleLaplace(n, scale);

		double mean = mean(samples);
		double theoreticalStdErr = scale * Math.sqrt(2.0) / Math.sqrt(n);
		Assert.assertTrue("Laplace mean should be near 0", Math.abs(mean) < 5 * theoreticalStdErr);

		double variance = variance(samples);
		double expected = 2.0 * scale * scale;
		Assert.assertEquals("Laplace variance", expected, variance, 0.1 * expected);
	}

	@Test
	public void testGaussianNoiseDistribution() throws ReflectiveOperationException {
		// Same idea as testLaplaceNoiseDistribution: check mean and variance from one sample set.
		int n = 10_000;
		double sigma = 2.0;
		double[] samples = sampleGaussian(n, sigma);

		double mean = mean(samples);
		double theoreticalStdErr = sigma / Math.sqrt(n);
		Assert.assertTrue("Gaussian mean should be near 0", Math.abs(mean) < 5 * theoreticalStdErr);

		double variance = variance(samples);
		double expected = sigma * sigma;
		Assert.assertEquals("Gaussian variance", expected, variance, 0.1 * expected);
	}

	// -----------------------------------------------------------------------
	// Helpers for noise distribution tests
	// -----------------------------------------------------------------------

	/** Sample n Laplace(0, scale) values via the fillLaplaceNoise method. */
	private static double[] sampleLaplace(int n, double scale) throws ReflectiveOperationException {
		Method m = DPBuiltinOps.class.getDeclaredMethod("fillLaplaceNoise", int.class, int.class, double.class);
		m.setAccessible(true);
		MatrixBlock block = (MatrixBlock) m.invoke(null, n, 1, scale);

		double[] out = new double[n];
		for(int i = 0; i < n; i++)
			out[i] = block.get(i, 0);
		return out;
	}

	/** Sample n N(0, sigma^2) values via the production fillGaussianNoise method. */
	private static double[] sampleGaussian(int n, double sigma) throws ReflectiveOperationException {
		Method m = DPBuiltinOps.class.getDeclaredMethod("fillGaussianNoise", int.class, int.class, double.class);
		m.setAccessible(true);
		MatrixBlock block = (MatrixBlock) m.invoke(null, n, 1, sigma);

		double[] out = new double[n];
		for(int i = 0; i < n; i++)
			out[i] = block.get(i, 0);
		return out;
	}

	private static double mean(double[] xs) {
		double s = 0;
		for(double x : xs)
			s += x;
		return s / xs.length;
	}

	private static double variance(double[] xs) {
		double m = mean(xs);
		double s = 0;
		for(double x : xs)
			s += (x - m) * (x - m);
		return s / (xs.length - 1);
	}
}
