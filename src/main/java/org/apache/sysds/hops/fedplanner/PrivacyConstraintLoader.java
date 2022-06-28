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

package org.apache.sysds.hops.fedplanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorkerHandlerException;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.propagation.PrivacyPropagator;
import org.apache.sysds.utils.JSONHelper;
import org.apache.wink.json4j.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

public class PrivacyConstraintLoader {

	private final Map<Long, Hop> memo = new HashMap<>();
	private final Map<String, Hop> transientWrites = new HashMap<>();

	public void loadConstraints(DMLProgram prog){
		rewriteStatementBlocks(prog, prog.getStatementBlocks(), null);
	}

	private void rewriteStatementBlocks(DMLProgram prog, List<StatementBlock> sbs, Map<String, Hop> paramMap) {
		sbs.forEach(block -> rewriteStatementBlock(prog, block, paramMap));
	}

	private void rewriteStatementBlock(DMLProgram prog, StatementBlock block, Map<String, Hop> paramMap){
		if(block instanceof WhileStatementBlock)
			rewriteWhileStatementBlock(prog, (WhileStatementBlock) block, paramMap);
		else if(block instanceof IfStatementBlock)
			rewriteIfStatementBlock(prog, (IfStatementBlock) block, paramMap);
		else if(block instanceof ForStatementBlock) {
			// This also includes ParForStatementBlocks
			rewriteForStatementBlock(prog, (ForStatementBlock) block, paramMap);
		}
		else if(block instanceof FunctionStatementBlock)
			rewriteFunctionStatementBlock(prog, (FunctionStatementBlock) block, paramMap);
		else {
			// StatementBlock type (no subclass)
			rewriteDefaultStatementBlock(prog, block, paramMap);
		}
	}

	private void rewriteWhileStatementBlock(DMLProgram prog, WhileStatementBlock whileSB, Map<String, Hop> paramMap) {
		Hop whilePredicateHop = whileSB.getPredicateHops();
		loadPrivacyConstraint(whilePredicateHop, paramMap);
		for(Statement stm : whileSB.getStatements()) {
			WhileStatement whileStm = (WhileStatement) stm;
			rewriteStatementBlocks(prog, whileStm.getBody(), paramMap);
		}
	}

	private void rewriteIfStatementBlock(DMLProgram prog, IfStatementBlock ifSB, Map<String, Hop> paramMap) {
		loadPrivacyConstraint(ifSB.getPredicateHops(), paramMap);
		for(Statement statement : ifSB.getStatements()) {
			IfStatement ifStatement = (IfStatement) statement;
			rewriteStatementBlocks(prog, ifStatement.getIfBody(), paramMap);
			rewriteStatementBlocks(prog, ifStatement.getElseBody(), paramMap);
		}
	}

	private void rewriteForStatementBlock(DMLProgram prog, ForStatementBlock forSB, Map<String, Hop> paramMap) {
		loadPrivacyConstraint(forSB.getFromHops(), paramMap);
		loadPrivacyConstraint(forSB.getToHops(), paramMap);
		loadPrivacyConstraint(forSB.getIncrementHops(), paramMap);
		for(Statement statement : forSB.getStatements()) {
			ForStatement forStatement = ((ForStatement) statement);
			rewriteStatementBlocks(prog, forStatement.getBody(), paramMap);
		}
	}

	private void rewriteFunctionStatementBlock(DMLProgram prog, FunctionStatementBlock funcSB, Map<String, Hop> paramMap) {
		for(Statement statement : funcSB.getStatements()) {
			FunctionStatement funcStm = (FunctionStatement) statement;
			rewriteStatementBlocks(prog, funcStm.getBody(), paramMap);
		}
	}

	private void rewriteDefaultStatementBlock(DMLProgram prog, StatementBlock sb, Map<String, Hop> paramMap) {
		if(sb.hasHops()) {
			for(Hop sbHop : sb.getHops()) {
				loadPrivacyConstraint(sbHop, paramMap);
				if(sbHop instanceof FunctionOp) {
					String funcName = ((FunctionOp) sbHop).getFunctionName();
					Map<String, Hop> funcParamMap = FederatedPlannerUtils.getParamMap((FunctionOp) sbHop);
					if ( paramMap != null && funcParamMap != null)
						funcParamMap.putAll(paramMap);
					paramMap = funcParamMap;
					FunctionStatementBlock sbFuncBlock = prog.getBuiltinFunctionDictionary().getFunction(funcName);
					rewriteStatementBlock(prog, sbFuncBlock, paramMap);
				}
			}
		}
	}

	private void loadPrivacyConstraint(Hop root, Map<String, Hop> paramMap){
		if ( root != null && !memo.containsKey(root.getHopID()) ){
			for ( Hop input : root.getInput() ){
				loadPrivacyConstraint(input, paramMap);
			}
			propagatePrivConstraintsLocal(root, paramMap);
			memo.put(root.getHopID(), root);
		}
	}

	private void propagatePrivConstraintsLocal(Hop currentHop, Map<String, Hop> paramMap){
		if ( currentHop.isFederatedDataOp() )
			loadFederatedPrivacyConstraints(currentHop);
		else if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTWRITE) ){
			currentHop.setPrivacy(currentHop.getInput(0).getPrivacy());
			transientWrites.put(currentHop.getName(), currentHop);
		}
		else if ( HopRewriteUtils.isData(currentHop, Types.OpOpData.TRANSIENTREAD) ){
			currentHop.setPrivacy(FederatedPlannerUtils.getTransientInputs(currentHop, paramMap, transientWrites).get(0).getPrivacy());
		} else {
			PrivacyPropagator.hopPropagation(currentHop);
		}
	}

	/**
	 * Get privacy constraints from federated workers for DataOps.
	 * @hop hop for which privacy constraints are loaded
	 */
	private static void loadFederatedPrivacyConstraints(Hop hop){
		try {
			PrivacyConstraint.PrivacyLevel constraintLevel = hop.getInput(0).getInput().stream().parallel()
				.map( in -> ((LiteralOp)in).getStringValue() )
				.map(PrivacyConstraintLoader::sendPrivConstraintRequest)
				.map(PrivacyConstraintLoader::unwrapPrivConstraint)
				.map(constraint -> (constraint != null) ? constraint.getPrivacyLevel() : PrivacyConstraint.PrivacyLevel.None)
				.reduce(PrivacyConstraint.PrivacyLevel.None, (out,in) -> {
					if ( out == PrivacyConstraint.PrivacyLevel.Private || in == PrivacyConstraint.PrivacyLevel.Private )
						return PrivacyConstraint.PrivacyLevel.Private;
					else if ( out == PrivacyConstraint.PrivacyLevel.PrivateAggregation || in == PrivacyConstraint.PrivacyLevel.PrivateAggregation )
						return PrivacyConstraint.PrivacyLevel.PrivateAggregation;
					else
						return out;
				});
			PrivacyConstraint fedDataPrivConstraint = (constraintLevel != PrivacyConstraint.PrivacyLevel.None) ?
				new PrivacyConstraint(constraintLevel) : null;

			hop.setPrivacy(fedDataPrivConstraint);
		}
		catch(Exception ex) {
			throw new DMLException(ex);
		}
	}

	private static Future<FederatedResponse> sendPrivConstraintRequest(String address)
	{
		try{
			String[] parsedAddress = InitFEDInstruction.parseURL(address);
			String host = parsedAddress[0];
			int port = Integer.parseInt(parsedAddress[1]);
			PrivacyConstraintRetriever retriever = new PrivacyConstraintRetriever(parsedAddress[2]);
			FederatedRequest privacyRetrieval =
				new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1, retriever);
			InetSocketAddress inetAddress = new InetSocketAddress(InetAddress.getByName(host), port);
			return FederatedData.executeFederatedOperation(inetAddress, privacyRetrieval);
		} catch(UnknownHostException ex){
			throw new DMLException(ex);
		}
	}

	private static PrivacyConstraint unwrapPrivConstraint(Future<FederatedResponse> privConstraintFuture)
	{
		try {
			FederatedResponse privConstraintResponse = privConstraintFuture.get();
			return (PrivacyConstraint) privConstraintResponse.getData()[0];
		} catch(Exception ex){
			throw new DMLException(ex);
		}
	}

	/**
	 * FederatedUDF for retrieving privacy constraint of data stored in file name.
	 */
	public static class PrivacyConstraintRetriever extends FederatedUDF {
		private static final long serialVersionUID = 3551741240135587183L;
		private final String filename;

		public PrivacyConstraintRetriever(String filename){
			super(new long[]{});
			this.filename = filename;
		}

		/**
		 * Reads metadata JSON object, parses privacy constraint and returns the constraint in FederatedResponse.
		 * @param ec execution context
		 * @param data one or many data objects
		 * @return FederatedResponse with privacy constraint object
		 */
		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			PrivacyConstraint privacyConstraint;
			FileSystem fs = null;
			try {
				String mtdname = DataExpression.getMTDFileName(filename);
				Path path = new Path(mtdname);
				fs = IOUtilFunctions.getFileSystem(mtdname);
				try(BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
					JSONObject metadataObject = JSONHelper.parse(br);
					privacyConstraint = PrivacyPropagator.parseAndReturnPrivacyConstraint(metadataObject);
				}
			}
			catch (DMLPrivacyException | FederatedWorkerHandlerException ex){
				throw ex;
			}
			catch (Exception ex) {
				String msg = "Exception in reading metadata of: " + filename;
				throw new DMLRuntimeException(msg);
			}
			finally {
				IOUtilFunctions.closeSilently(fs);
			}
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, privacyConstraint);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

}
