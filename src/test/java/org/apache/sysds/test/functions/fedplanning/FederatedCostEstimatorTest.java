package org.apache.sysds.test.functions.fedplanning;
///*
// * Licensed to the Apache Software Foundation (ASF) under one
// * or more contributor license agreements.  See the NOTICE file
// * distributed with this work for additional information
// * regarding copyright ownership.  The ASF licenses this file
// * to you under the Apache License, Version 2.0 (the
// * "License"); you may not use this file except in compliance
// * with the License.  You may obtain a copy of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package org.apache.sysds.test.functions.privacy.fedplanning;
//
//import net.jcip.annotations.NotThreadSafe;
//import org.apache.sysds.api.DMLScript;
//import org.apache.sysds.common.Types;
//import org.apache.sysds.conf.ConfigurationManager;
//import org.apache.sysds.conf.DMLConfig;
//import org.apache.sysds.hops.AggBinaryOp;
//import org.apache.sysds.hops.BinaryOp;
//import org.apache.sysds.hops.DataOp;
//import org.apache.sysds.hops.Hop;
//import org.apache.sysds.hops.LiteralOp;
//import org.apache.sysds.hops.NaryOp;
//import org.apache.sysds.hops.ReorgOp;
//import org.apache.sysds.hops.cost.FederatedCost;
//import org.apache.sysds.hops.cost.FederatedCostEstimator;
//import org.apache.sysds.hops.fedplanner.FederatedPlannerCostbased;
//import org.apache.sysds.hops.ipa.FunctionCallGraph;
//import org.apache.sysds.parser.DMLProgram;
//import org.apache.sysds.parser.DMLTranslator;
//import org.apache.sysds.parser.LanguageException;
//import org.apache.sysds.parser.ParserFactory;
//import org.apache.sysds.parser.ParserWrapper;
//import org.apache.sysds.parser.StatementBlock;
//import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
//import org.apache.sysds.test.AutomatedTestBase;
//import org.apache.sysds.test.TestConfiguration;
//import org.junit.After;
//import org.junit.Assert;
//import org.junit.Before;
//import org.junit.BeforeClass;
//import org.junit.Test;
//
//import java.io.FileNotFoundException;
//import java.io.IOException;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Set;
//
//import static org.apache.sysds.common.Types.OpOp2.MULT;
//
//@NotThreadSafe
//public class FederatedCostEstimatorTest extends AutomatedTestBase {
//
//	private static final String TEST_DIR = "functions/privacy/fedplanning/";
//	private static final String HOME = SCRIPT_DIR + TEST_DIR;
//	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedCostEstimatorTest.class.getSimpleName() + "/";
//	FederatedCostEstimator fedCostEstimator = new FederatedCostEstimator();
//
//	private static double COMPUTE_FLOPS;
//	private static double READ_PS;
//	private static double NETWORK_PS;
//
//	@Override
//	public void setUp() {}
//
//	@BeforeClass
//	public static void storeConstants(){
//		COMPUTE_FLOPS = FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS;
//		READ_PS = FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS;
//		NETWORK_PS = FederatedCostEstimator.WORKER_NETWORK_BANDWIDTH_BYTES_PS;
//	}
//
//	@Before
//	public void setConstants(){
//		FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS = 2;
//		FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS = 10;
//		FederatedCostEstimator.WORKER_NETWORK_BANDWIDTH_BYTES_PS = 5;
//	}
//
//	@After
//	public void resetConstants(){
//		FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS = COMPUTE_FLOPS;
//		FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS = READ_PS;
//		FederatedCostEstimator.WORKER_NETWORK_BANDWIDTH_BYTES_PS = NETWORK_PS;
//	}
//
//	@Test
//	public void simpleBinary() {
//
//		/*
//		 * HOP			Occurences		ComputeCost		ReadCost	ComputeCostFinal	ReadCostFinal
//		 * ------------------------------------------------------------------------------------------
//		 * LiteralOp	16				1				0			0.0625				0
//		 * DataGenOp	2				100				64			6.25				6.4
//		 * BinaryOp		1				100				1600		6.25				160
//		 * TOSTRING		1				1				800			0.0625				80
//		 * UnaryOp		1				1				8			0.0625				0.8
//		 */
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//
//		double expectedCost = computeCost + readCost;
//		runTest("BinaryCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void simpleBinaryHopRelTest() {
//		runHopRelTest("BinaryCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void ifElseTest(){
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//		double expectedCost = ((computeCost + readCost + 0.8 + 0.0625 + 0.0625) / 2) + 0.0625 + 0.8 + 0.0625;
//		runTest("IfElseCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void ifElseHopRelTest(){
//		runHopRelTest("IfElseCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void whileTest(){
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//		double expectedCost = (computeCost + readCost + 0.0625 + 0.0625 + 0.8) * StatementBlock.DEFAULT_LOOP_REPETITIONS;
//		runTest("WhileCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void whileHopRelTest(){
//		runHopRelTest("WhileCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void forLoopTest(){
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//		double predicateCost = 0.0625 + 0.8 + 0.0625 + 0.0625 + 0.8 + 0.0625 + 0.0625 + 0.8 + 0.0625;
//		double expectedCost = (computeCost + readCost + predicateCost) * 5;
//		runTest("ForLoopCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void forLoopHopRelTest(){
//		runHopRelTest("ForLoopCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void parForLoopTest(){
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//		double predicateCost = 0.0625 + 0.8 + 0.0625 + 0.0625 + 0.8 + 0.0625 + 0.0625 + 0.8 + 0.0625;
//		double expectedCost = (computeCost + readCost + predicateCost) * 5;
//		runTest("ParForLoopCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void parForLoopHopRelTest(){
//		runHopRelTest("ParForLoopCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void functionTest(){
//		double computeCost = (16+2*100+100+1+1) / (FederatedCostEstimator.WORKER_COMPUTE_BANDWIDTH_FLOPS * FederatedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
//		double readCost = (2*64+1600+800+8) / (FederatedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
//		double expectedCost = (computeCost + readCost);
//		runTest("FunctionCostEstimatorTest.dml", false, expectedCost);
//	}
//
//	@Test
//	public void functionHopRelTest(){
//		runHopRelTest("FunctionCostEstimatorTest.dml", false);
//	}
//
//	@Test
//	public void federatedMultiply() {
//
//		double literalOpCost = 10*0.0625;
//		double naryOpCostSpecial = (0.125+2.2);
//		double naryOpCostSpecial2 = (0.25+6.4);
//		double naryOpCost = 4*(0.125+1.6);
//		double reorgOpCost = 6250+80015.2+160030.4;
//		double binaryOpMultCost = 3125+160000;
//		double aggBinaryOpCost = 125000+160015.2+160030.4+190.4;
//		double dataOpCost = 2*(6250+5.6);
//		double dataOpWriteCost = 6.25+100.3;
//
//		double expectedCost = literalOpCost + naryOpCost + naryOpCostSpecial + naryOpCostSpecial2 + reorgOpCost
//			+ binaryOpMultCost + aggBinaryOpCost + dataOpCost + dataOpWriteCost;
//		runTest("FederatedMultiplyCostEstimatorTest.dml", false, expectedCost);
//
//		double aggBinaryActualCost = hops.stream()
//			.filter(hop -> hop instanceof AggBinaryOp)
//			.mapToDouble(aggHop -> aggHop.getFederatedCost().getTotal()-aggHop.getFederatedCost().getInputTotalCost())
//			.sum();
//		Assert.assertEquals(aggBinaryOpCost, aggBinaryActualCost, 0.0001);
//
//		double writeActualCost = hops.stream()
//			.filter(hop -> hop instanceof DataOp)
//			.mapToDouble(writeHop -> writeHop.getFederatedCost().getTotal()-writeHop.getFederatedCost().getInputTotalCost())
//			.sum();
//		Assert.assertEquals(dataOpWriteCost+dataOpCost, writeActualCost, 0.0001);
//	}
//
//	Set<Hop> hops = new HashSet<>();
//
//	/**
//	 * Recursively adds the hop and its inputs to the set of hops.
//	 * @param hop root to be added to set of hops
//	 */
//	private void addHop(Hop hop){
//		hops.add(hop);
//		for(Hop inHop : hop.getInput()){
//			addHop(inHop);
//		}
//	}
//
//	/**
//	 * Sets dimensions of federated X and Y and sets binary multiplication to FOUT.
//	 * @param prog dml program where the HOPS are modified
//	 */
//	private void modifyFedouts(DMLProgram prog){
//		prog.getStatementBlocks().forEach(stmBlock -> stmBlock.getHops().forEach(this::addHop));
//		hops.forEach(hop -> {
//			if ( hop instanceof DataOp || (hop instanceof BinaryOp && ((BinaryOp) hop).getOp() == MULT ) ){
//				hop.setFederatedOutput(FEDInstruction.FederatedOutput.FOUT);
//				hop.setExecType(Types.ExecType.FED);
//			} else {
//				hop.setFederatedOutput(FEDInstruction.FederatedOutput.LOUT);
//			}
//			if ( hop.getOpString().equals("Fed Y") || hop.getOpString().equals("Fed X") ){
//				hop.setDim1(10000);
//				hop.setDim2(10);
//			}
//		});
//	}
//
//	@SuppressWarnings("unused")
//	private void printHopsInfo(){
//		//LiteralOp
//		long literalCount = hops.stream().filter(hop -> hop instanceof LiteralOp).count();
//		System.out.println("LiteralOp Count: " + literalCount);
//		//NaryOp
//		long naryCount = hops.stream().filter(hop -> hop instanceof NaryOp).count();
//		System.out.println("NaryOp Count " + naryCount);
//		//ReorgOp
//		long reorgCount = hops.stream().filter(hop -> hop instanceof ReorgOp).count();
//		System.out.println("ReorgOp Count: " + reorgCount);
//		//BinaryOp
//		long binaryCount = hops.stream().filter(hop -> hop instanceof BinaryOp).count();
//		System.out.println("Binary count: " + binaryCount);
//		//AggBinaryOp
//		long aggBinaryCount = hops.stream().filter(hop -> hop instanceof AggBinaryOp).count();
//		System.out.println("AggBinaryOp Count: " + aggBinaryCount);
//		//DataOp
//		long dataOpCount = hops.stream().filter(hop -> hop instanceof DataOp).count();
//		System.out.println("DataOp Count: " + dataOpCount);
//
//		hops.stream().map(Hop::getClass).distinct().forEach(System.out::println);
//	}
//
//	private DMLProgram testSetup(String scriptFilename) throws IOException{
//		setTestConfig(scriptFilename);
//		String dmlScriptString = readScript(scriptFilename);
//
//		//parsing, dependency analysis and constructing hops (step 3 and 4 in DMLScript.java)
//		ParserWrapper parser = ParserFactory.createParser();
//		DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
//		DMLTranslator dmlt = new DMLTranslator(prog);
//		dmlt.liveVariableAnalysis(prog);
//		dmlt.validateParseTree(prog);
//		dmlt.constructHops(prog);
//		if ( scriptFilename.equals("FederatedMultiplyCostEstimatorTest.dml")){
//			modifyFedouts(prog);
//			dmlt.rewriteHopsDAG(prog);
//			hops = new HashSet<>();
//			prog.getStatementBlocks().forEach(stmBlock -> stmBlock.getHops().forEach(this::addHop));
//		}
//		return prog;
//	}
//
//	private void compareResults(DMLProgram prog) {
//		FederatedPlannerCostbased rewriter = new FederatedPlannerCostbased();
//		rewriter.rewriteProgram(prog, new FunctionCallGraph(prog), null);
//
//		double actualCost = 0;
//		for ( Hop root : rewriter.getTerminalHops() ){
//			actualCost += root.getFederatedCost().getTotal();
//		}
//
//
//		rewriter.getTerminalHops().forEach(Hop::resetFederatedCost);
//		fedCostEstimator = new FederatedCostEstimator();
//		double expectedCost = 0;
//		for ( Hop root : rewriter.getTerminalHops() )
//			expectedCost += fedCostEstimator.costEstimate(root).getTotal();
//		Assert.assertEquals(expectedCost, actualCost, 0.0001);
//	}
//
//	private void runHopRelTest( String scriptFilename, boolean expectedException ) {
//		boolean raisedException = false;
//		try
//		{
//			DMLProgram prog = testSetup(scriptFilename);
//			compareResults(prog);
//		}
//		catch(LanguageException ex) {
//			raisedException = true;
//			if(raisedException!=expectedException)
//				ex.printStackTrace();
//		}
//		catch(Exception ex2) {
//			ex2.printStackTrace();
//			throw new RuntimeException(ex2);
//		}
//
//		Assert.assertEquals("Expected exception does not match raised exception",
//			expectedException, raisedException);
//	}
//
//	private void runTest( String scriptFilename, boolean expectedException, double expectedCost ) {
//		boolean raisedException = false;
//		try
//		{
//			DMLProgram prog = testSetup(scriptFilename);
//
//			fedCostEstimator = new FederatedCostEstimator();
//			FederatedCost actualCost = fedCostEstimator.costEstimate(prog);
//			Assert.assertEquals(expectedCost, actualCost.getTotal(), 0.0001);
//		}
//		catch(LanguageException ex) {
//			raisedException = true;
//			if(raisedException!=expectedException)
//				ex.printStackTrace();
//		}
//		catch(Exception ex2) {
//			ex2.printStackTrace();
//			throw new RuntimeException(ex2);
//		}
//
//		Assert.assertEquals("Expected exception does not match raised exception",
//			expectedException, raisedException);
//	}
//
//	private void setTestConfig(String scriptFilename) throws FileNotFoundException {
//		int index = scriptFilename.lastIndexOf(".dml");
//		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
//		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
//		addTestConfiguration(testName, testConfig);
//		loadTestConfiguration(testConfig);
//
//		DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
//		ConfigurationManager.setLocalConfig(conf);
//	}
//
//	private static String readScript(String scriptFilename) throws IOException {
//		return DMLScript.readDMLScript(true, HOME + scriptFilename);
//	}
//}
