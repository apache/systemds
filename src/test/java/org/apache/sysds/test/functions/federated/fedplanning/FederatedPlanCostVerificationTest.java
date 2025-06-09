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

package org.apache.sysds.test.functions.federated.fedplanning;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.FedPlan;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEstimator;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * Tests for verifying that the total cost of the optimal federated plan
 * matches the sum of individually calculated costs for all nodes in the plan.
 * This test uses bottom-up DFS traversal to calculate costs.
 */
public class FederatedPlanCostVerificationTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedPlanCostVerificationTest.class.getName());

	private final static String TEST_DIR = "functions/privacy/fedplanning/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedPlanCostVerificationTest.class.getSimpleName()
			+ "/";
	private static File TEST_CONF_FILE;

	private final static int blocksize = 1024;
	public final int rows = 1000;
	public final int cols = 100;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("FederatedKMeansPlanningTest",
				new TestConfiguration(TEST_CLASS_DIR, "FederatedKMeansPlanningTest", new String[] { "Z" }));
		addTestConfiguration("FederatedL2SVMPlanningTest",
				new TestConfiguration(TEST_CLASS_DIR, "FederatedL2SVMPlanningTest", new String[] { "Z" }));
	}

	@Test
	public void testKMeansCostVerification() {
		runCostVerificationTest("FederatedKMeansPlanningTest", true);
	}

	@Test
	public void testL2SVMCostVerification() {
		runCostVerificationTest("FederatedL2SVMPlanningTest", false);
	}

	@Test
	public void testKMeansCostVerificationWithPrivacy() {
		runCostVerificationTestWithPrivacy("FederatedKMeansPlanningTest", true, "private");
	}

	@Test
	public void testL2SVMCostVerificationWithPrivacy() {
		runCostVerificationTestWithPrivacy("FederatedL2SVMPlanningTest", false, "private-aggregate");
	}

	@Test
	public void testEmptyPlanCostVerification() {
		// Test edge case: empty plan
		FedPlan emptyPlan = createEmptyPlan();
		FederatedMemoTable emptyMemoTable = new FederatedMemoTable();
		
		double cost = calculateTotalCostBottomUpDFS(emptyPlan, emptyMemoTable);
		Assert.assertEquals("Empty plan should have zero cost", 0.0, cost, 0.0001);
	}

	@Test
	public void testNullInputHandling() {
		// Test edge case: null inputs
		double cost1 = calculateTotalCostBottomUpDFS(null, new FederatedMemoTable());
		Assert.assertEquals("Null plan should return zero cost", 0.0, cost1, 0.0001);
		
		FedPlan emptyPlan = createEmptyPlan();
		double cost2 = calculateTotalCostBottomUpDFS(emptyPlan, null);
		Assert.assertEquals("Null memo table should return zero cost", 0.0, cost2, 0.0001);
	}

	private FedPlan createEmptyPlan() {
		// Create a mock empty plan for testing
		return new FedPlan(0.0, null, new ArrayList<>());
	}

	private void runCostVerificationTest(String testName, boolean isKMeans) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null;

		try {
			// Setup configuration for cost-based planning
			TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-cost-based.xml");
			getAndLoadTestConfiguration(testName);

			// Configure cost-based planner
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.FEDERATED_PLANNER, "compile_cost_based");

			String HOME = SCRIPT_DIR + TEST_DIR;

			// Write input matrices
			if (isKMeans) {
				writeKMeansInputMatrices();
			} else {
				writeL2SVMInputMatrices();
			}

			// Start federated workers
			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);

			// Read and parse DML script
			fullDMLScriptName = HOME + testName + ".dml";
			String dmlScriptString = DMLScript.readDMLScript(true, fullDMLScriptName);

			// Parse and construct Hop DAG using nvargs like the original tests
			ParserWrapper parser = ParserFactory.createParser();
			
			// Set up nvargs like the original tests do
			Map<String, String> nvargs = new HashMap<>();
			nvargs.put("X1", TestUtils.federatedAddress(port1, input("X1")));
			nvargs.put("X2", TestUtils.federatedAddress(port2, input("X2")));
			if (!isKMeans) {
				nvargs.put("Y", input("Y"));
			}
			nvargs.put("r", String.valueOf(rows));
			nvargs.put("c", String.valueOf(cols));
			nvargs.put("Z", output("Z"));
			
			// Debug: log nvargs
			LOG.info("nvargs: " + nvargs);
			
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, nvargs);
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);

			// Create memo table and enumerate federated plans
			FederatedMemoTable memoTable = new FederatedMemoTable();
			FedPlan optimalPlan = FederatedPlanCostEnumerator.enumerateProgram(prog,
					memoTable, false);

			// Verify cost calculation
			double reportedTotalCost = optimalPlan.getCumulativeCost();
			double calculatedTotalCost = calculateTotalCostBottomUpDFS(optimalPlan, memoTable);

			// Log the costs for debugging
			LOG.info("Reported total cost: " + reportedTotalCost);
			LOG.info("Calculated total cost: " + calculatedTotalCost);

			// Assert that costs match with improved delta calculation
			double absoluteDelta = 0.0001;
			double relativeDelta = Math.max(Math.abs(reportedTotalCost), Math.abs(calculatedTotalCost)) * 0.001;
			double finalDelta = Math.max(absoluteDelta, relativeDelta);
			
			// Additional validation for edge cases
			if (Double.isNaN(reportedTotalCost) || Double.isInfinite(reportedTotalCost)) {
				Assert.fail("Reported total cost is invalid: " + reportedTotalCost);
			}
			if (Double.isNaN(calculatedTotalCost) || Double.isInfinite(calculatedTotalCost)) {
				Assert.fail("Calculated total cost is invalid: " + calculatedTotalCost);
			}
			
			Assert.assertEquals("Optimal plan cost should match sum of individual node costs",
					reportedTotalCost, calculatedTotalCost, finalDelta);

		} catch (Exception e) {
			e.printStackTrace();
			Assert.fail(e.getMessage());
		} finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void runCostVerificationTestWithPrivacy(String testName, boolean isKMeans, String privacyConstraints) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		Thread t1 = null, t2 = null;

		try {
			// Setup configuration for cost-based planning
			TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-cost-based.xml");
			getAndLoadTestConfiguration(testName);

			// Configure cost-based planner
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.FEDERATED_PLANNER, "compile_cost_based");

			String HOME = SCRIPT_DIR + TEST_DIR;

			// Write input matrices with privacy constraints
			if (isKMeans) {
				writeKMeansInputMatricesWithPrivacy(privacyConstraints);
			} else {
				writeL2SVMInputMatricesWithPrivacy(privacyConstraints);
			}

			// Start federated workers
			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);

			// Read and parse DML script
			fullDMLScriptName = HOME + testName + ".dml";
			String dmlScriptString = DMLScript.readDMLScript(true, fullDMLScriptName);

			// Set up federated addresses in the script
			dmlScriptString = dmlScriptString.replace("$X1", TestUtils.federatedAddress(port1, input("X1")));
			dmlScriptString = dmlScriptString.replace("$X2", TestUtils.federatedAddress(port2, input("X2")));
			dmlScriptString = dmlScriptString.replace("$Y", input("Y"));
			dmlScriptString = dmlScriptString.replace("$r", String.valueOf(rows));
			dmlScriptString = dmlScriptString.replace("$c", String.valueOf(cols));
			dmlScriptString = dmlScriptString.replace("$Z", output("Z"));

			// Parse and construct Hop DAG
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);

			// Create memo table and enumerate federated plans
			FederatedMemoTable memoTable = new FederatedMemoTable();
			FedPlan optimalPlan = FederatedPlanCostEnumerator.enumerateProgram(prog,
					memoTable, false);

			// Verify cost calculation
			double reportedTotalCost = optimalPlan.getCumulativeCost();
			double calculatedTotalCost = calculateTotalCostBottomUpDFS(optimalPlan, memoTable);

			// Log the costs for debugging
			LOG.info("Reported total cost with " + privacyConstraints + ": " + reportedTotalCost);
			LOG.info("Calculated total cost with " + privacyConstraints + ": " + calculatedTotalCost);

			// Assert that costs match within a small delta (for floating point comparison)
			double delta = 0.0001;
			Assert.assertEquals("Optimal plan cost should match sum of individual node costs",
					reportedTotalCost, calculatedTotalCost, delta);

		} catch (Exception e) {
			e.printStackTrace();
			Assert.fail(e.getMessage());
		} finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	/**
	 * Calculates the total cost using bottom-up DFS traversal.
	 * This method performs a post-order traversal to ensure child costs
	 * are calculated before parent costs.
	 * 
	 * @param rootPlan  The root of the optimal federated plan
	 * @param memoTable The federated memo table containing plan information
	 * @return The total calculated cost
	 */
	private double calculateTotalCostBottomUpDFS(FedPlan rootPlan,
			FederatedMemoTable memoTable) {

		// Edge case: null inputs
		if (rootPlan == null || memoTable == null) {
			LOG.warn("Null input detected: rootPlan=" + rootPlan + ", memoTable=" + memoTable);
			return 0.0;
		}

		// Edge case: empty root plan
		if (rootPlan.getChildFedPlans() == null || rootPlan.getChildFedPlans().isEmpty()) {
			LOG.warn("Root plan has no children - this might be an empty plan");
			return 0.0;
		}

		// Map to store calculated costs for each node
		Map<Pair<Long, FederatedOutput>, Double> nodeCosts = new HashMap<>();

		// Set to track visited nodes during DFS
		Set<Pair<Long, FederatedOutput>> visited = new HashSet<>();

		// Set to track nodes currently being processed (for cycle detection)
		Set<Pair<Long, FederatedOutput>> processing = new HashSet<>();

		// Stack for DFS traversal
		Stack<Pair<FedPlan, Boolean>> dfsStack = new Stack<>();

		// Timeout handling
		long startTime = System.currentTimeMillis();
		long timeoutMs = 30000; // 30 seconds
		int nodeCount = 0;
		final int MAX_NODES = 10000; // Prevent excessive memory usage

		// Start DFS from root's children (root is dummy node)
		for (Pair<Long, FederatedOutput> childPlanPair : rootPlan.getChildFedPlans()) {
			if (childPlanPair == null) {
				LOG.warn("Null child plan pair detected in root");
				continue;
			}

			FedPlan childPlan = memoTable.getFedPlanAfterPrune(childPlanPair);
			if (childPlan != null) {
				dfsStack.push(new ImmutablePair<>(childPlan, false));
			} else {
				LOG.warn("Could not retrieve child plan for: " + childPlanPair);
			}
		}

		// Perform bottom-up DFS traversal
		while (!dfsStack.isEmpty()) {
			// Timeout check
			if (System.currentTimeMillis() - startTime > timeoutMs) {
				throw new RuntimeException("Cost calculation timeout after " + timeoutMs + "ms");
			}

			// Node count check
			if (nodeCount > MAX_NODES) {
				throw new RuntimeException("Too many nodes processed: " + nodeCount + " > " + MAX_NODES);
			}

			Pair<FedPlan, Boolean> current = dfsStack.pop();
			FedPlan currentPlan = current.getLeft();
			boolean isPostOrder = current.getRight();

			// Additional null check
			if (currentPlan == null) {
				LOG.warn("Null current plan detected during traversal");
				continue;
			}

			Pair<Long, FederatedOutput> currentNodeKey = new ImmutablePair<>(currentPlan.getHopID(),
					currentPlan.getFedOutType());

			if (isPostOrder) {
				// Post-order visit: calculate cost for this node
				if (!nodeCosts.containsKey(currentNodeKey)) {
					// Remove from processing set
					processing.remove(currentNodeKey);

					double nodeCost = calculateNodeCost(currentPlan, memoTable, nodeCosts);

					// Edge case: check for invalid costs
					if (Double.isNaN(nodeCost) || Double.isInfinite(nodeCost)) {
						LOG.warn("Invalid cost calculated for node " + currentNodeKey + ": " + nodeCost);
						nodeCost = 0.0; // Default to 0 for invalid costs
					}

					nodeCosts.put(currentNodeKey, nodeCost);
					
					LOG.debug("Node " + currentNodeKey + ": cost=" + nodeCost);
				}
			} else {
				// Pre-order visit: schedule post-order visit and visit children
				if (!visited.contains(currentNodeKey)) {
					// Edge case: cycle detection
					if (processing.contains(currentNodeKey)) {
						LOG.warn("Cycle detected at node: " + currentNodeKey + " - skipping to avoid infinite loop");
						continue;
					}

					visited.add(currentNodeKey);
					processing.add(currentNodeKey);
					nodeCount++;

					// Schedule post-order visit for this node
					dfsStack.push(new ImmutablePair<>(currentPlan, true));

					// Schedule visits for all children
					if (currentPlan.getChildFedPlans() != null) {
						for (Pair<Long, FederatedOutput> childPlanPair : currentPlan.getChildFedPlans()) {
							if (childPlanPair == null) {
								LOG.warn("Null child plan pair detected");
								continue;
							}

							FedPlan childPlan = memoTable.getFedPlanAfterPrune(childPlanPair);
							if (childPlan != null) {
								Pair<Long, FederatedOutput> childNodeKey = new ImmutablePair<>(childPlan.getHopID(),
										childPlan.getFedOutType());
								if (!visited.contains(childNodeKey) && !processing.contains(childNodeKey)) {
									dfsStack.push(new ImmutablePair<>(childPlan, false));
								}
							}
						}
					}
				}
			}
		}

		// Calculate total cost from root's children
		double totalCost = 0.0;
		for (Pair<Long, FederatedOutput> childPlanPair : rootPlan.getChildFedPlans()) {
			if (childPlanPair == null) continue;

			Double childCost = nodeCosts.get(childPlanPair);
			if (childCost != null) {
				// Edge case: check for valid costs before adding
				if (!Double.isNaN(childCost) && !Double.isInfinite(childCost)) {
					totalCost += childCost;
				} else {
					LOG.warn("Invalid child cost detected: " + childCost + " for " + childPlanPair);
				}
			} else {
				LOG.warn("No cost calculated for child: " + childPlanPair);
			}
		}

		// Final validation
		if (Double.isNaN(totalCost) || Double.isInfinite(totalCost)) {
			LOG.warn("Invalid total cost calculated: " + totalCost);
			return 0.0;
		}

		LOG.info("DFS completed: processed " + nodeCount + " nodes in " + 
		         (System.currentTimeMillis() - startTime) + "ms");

		return totalCost;
	}

	/**
	 * Calculates the cost for a single node including its self cost and
	 * the costs from its children.
	 */
	private double calculateNodeCost(FedPlan plan,
			FederatedMemoTable memoTable, Map<Pair<Long, FederatedOutput>, Double> nodeCosts) {

		// Null check for plan
		if (plan == null) {
			LOG.warn("Null plan provided to calculateNodeCost");
			return 0.0;
		}

		// Get the hop common for this plan
		Pair<Long, FederatedOutput> nodeKey = new ImmutablePair<>(plan.getHopID(), plan.getFedOutType());
		FederatedMemoTable.FedPlanVariants variants = memoTable.getFedPlanVariants(nodeKey);

		if (variants == null) {
			LOG.warn("No variants found for node: " + nodeKey);
			return 0.0;
		}

		// Use the plan's built-in methods instead of accessing hopCommon directly
		double selfCost = 0.0;
		try {
			selfCost = plan.getSelfCost();
			
			// Validate self cost
			if (Double.isNaN(selfCost) || Double.isInfinite(selfCost) || selfCost < 0) {
				LOG.warn("Invalid self cost for node " + nodeKey + ": " + selfCost);
				selfCost = 0.0;
			}
		} catch (Exception e) {
			LOG.warn("Error getting self cost for node " + nodeKey + ": " + e.getMessage());
			selfCost = 0.0;
		}

		// Apply compute weight (for loops/conditions)
		double computeWeight = 1.0;
		try {
			computeWeight = plan.getComputeWeight();
			if (Double.isNaN(computeWeight) || Double.isInfinite(computeWeight) || computeWeight <= 0) {
				LOG.warn("Invalid compute weight for node " + nodeKey + ": " + computeWeight + ", using 1.0");
				computeWeight = 1.0;
			}
		} catch (Exception e) {
			LOG.warn("Error getting compute weight for node " + nodeKey + ": " + e.getMessage());
			computeWeight = 1.0;
		}
		
		double weightedSelfCost = selfCost * computeWeight;

		// Account for parent sharing - we'll estimate this from the plan structure
		// Since we can't access numParents directly, we'll use a simple approach
		double finalSelfCost = weightedSelfCost; // For now, don't divide by parents

		// Add costs from children
		double childrenCost = 0.0;
		
		// Null check for child plans
		if (plan.getChildFedPlans() != null) {
			for (Pair<Long, FederatedOutput> childPlanPair : plan.getChildFedPlans()) {
				if (childPlanPair == null) {
					LOG.warn("Null child plan pair in node: " + nodeKey);
					continue;
				}

				// Get child's cumulative cost (already calculated in bottom-up traversal)
				Double childCumulativeCost = nodeCosts.get(childPlanPair);
				if (childCumulativeCost != null) {
					// Validate child cost
					if (!Double.isNaN(childCumulativeCost) && !Double.isInfinite(childCumulativeCost) && childCumulativeCost >= 0) {
						childrenCost += childCumulativeCost;
					} else {
						LOG.warn("Invalid child cumulative cost: " + childCumulativeCost + " for " + childPlanPair);
					}
				}

				// Add forwarding cost if federation status changes
				try {
					FedPlan childPlan = memoTable.getFedPlanAfterPrune(childPlanPair);
					if (childPlan != null && plan.getFedOutType() != childPlan.getFedOutType()) {
						double forwardingCost = childPlan.getForwardingCostPerParents();
						double forwardingWeight = plan.getChildForwardingWeight(childPlan.getLoopContext());

						// Validate forwarding cost and weight
						if (Double.isNaN(forwardingCost) || Double.isInfinite(forwardingCost) || forwardingCost < 0) {
							LOG.warn("Invalid forwarding cost: " + forwardingCost + " for " + childPlanPair);
							forwardingCost = 0.0;
						}
						
						if (Double.isNaN(forwardingWeight) || Double.isInfinite(forwardingWeight) || forwardingWeight < 0) {
							LOG.warn("Invalid forwarding weight: " + forwardingWeight + " for " + childPlanPair);
							forwardingWeight = 1.0;
						}

						childrenCost += forwardingCost * forwardingWeight;
					}
				} catch (Exception e) {
					LOG.warn("Error calculating forwarding cost for child " + childPlanPair + ": " + e.getMessage());
				}
			}
		}

		double totalNodeCost = finalSelfCost + childrenCost;

		// Final validation
		if (Double.isNaN(totalNodeCost) || Double.isInfinite(totalNodeCost) || totalNodeCost < 0) {
			LOG.warn("Invalid total node cost for " + nodeKey + ": " + totalNodeCost + 
			         " (selfCost=" + finalSelfCost + ", childrenCost=" + childrenCost + ")");
			return 0.0;
		}

		return totalNodeCost;
	}

	// Helper methods for writing input matrices
	private void writeKMeansInputMatrices() {
		writeStandardRowFedMatrix("X1", 65);
		writeStandardRowFedMatrix("X2", 75);
	}

	private void writeKMeansInputMatricesWithPrivacy(String privacyConstraints) {
		writeStandardRowFedMatrix("X1", 65, privacyConstraints);
		writeStandardRowFedMatrix("X2", 75, privacyConstraints);
	}

	private void writeL2SVMInputMatrices() {
		writeStandardRowFedMatrix("X1", 65);
		writeStandardRowFedMatrix("X2", 75);
		writeBinaryVector("Y", 44);
	}

	private void writeL2SVMInputMatricesWithPrivacy(String privacyConstraints) {
		writeStandardRowFedMatrix("X1", 65, privacyConstraints);
		writeStandardRowFedMatrix("X2", 75, privacyConstraints);
		writeBinaryVector("Y", 44);
	}

	private void writeBinaryVector(String matrixName, long seed) {
		double[][] matrix = getRandomMatrix(rows, 1, -1, 1, 1, seed);
		for (int i = 0; i < rows; i++)
			matrix[i][0] = (matrix[i][0] > 0) ? 1 : -1;
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, 1, blocksize, rows);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed) {
		int halfRows = rows / 2;
		writeStandardMatrix(matrixName, seed, halfRows);
	}

	private void writeStandardRowFedMatrix(String matrixName, long seed, String privacyConstraints) {
		int halfRows = rows / 2;
		writeStandardMatrix(matrixName, seed, halfRows, privacyConstraints);
	}

	private void writeStandardMatrix(String matrixName, long seed, int numRows) {
		double[][] matrix = getRandomMatrix(numRows, cols, 0, 1, 1, seed);
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc);
	}

	private void writeStandardMatrix(String matrixName, long seed, int numRows, String privacyConstraints) {
		double[][] matrix = getRandomMatrix(numRows, cols, 0, 1, 1, seed);
		MatrixCharacteristics mc = new MatrixCharacteristics(numRows, cols, blocksize, (long) numRows * cols);
		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraints);
	}

	@Override
	protected File getConfigTemplateFile() {
		// Use custom configuration file if set
		if (TEST_CONF_FILE != null) {
			LOG.info("Using custom configuration: " + TEST_CONF_FILE.getPath());
			return TEST_CONF_FILE;
		}
		return super.getConfigTemplateFile();
	}
}