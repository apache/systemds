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

package org.apache.sysds.hops.rewrite;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorkerHandlerException;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.propagation.PrivacyPropagator;
import org.apache.sysds.utils.JSONHelper;
import org.apache.wink.json4j.JSONObject;

import javax.net.ssl.SSLException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.Future;

public class RewriteFederatedExecution extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if ( roots == null )
			return null;
		for ( Hop root : roots )
			visitHop(root);

		return selectFederatedExecutionPlan(roots);
	}

	@Override public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		return null;
	}

	/**
	 * Select federated execution plan for every Hop in the DAG starting from given roots.
	 * @param roots starting point for going through the Hop DAG to update the FederatedOutput fields.
	 * @return the list of roots with updated FederatedOutput fields.
	 */
	private ArrayList<Hop> selectFederatedExecutionPlan(ArrayList<Hop> roots){
		for (Hop root : roots){
			root.resetVisitStatus();
		}
		for ( Hop root : roots ){
			visitFedPlanHop(root);
		}
		return roots;
	}

	/**
	 * Go through the Hop DAG and set the FederatedOutput field for each Hop from leaf to given currentHop.
	 * @param currentHop the Hop from which the DAG is visited
	 */
	private void visitFedPlanHop(Hop currentHop){
		if ( currentHop.isVisited() )
			return;
		if ( currentHop.getInput() != null && currentHop.getInput().size() > 0 && !isFederatedDataOp(currentHop) ){
			// Depth first to get to the input
			for ( Hop input : currentHop.getInput() )
				visitFedPlanHop(input);
		} else if ( isFederatedDataOp(currentHop) ) {
			// leaf federated node
			//TODO: This will block the cases where the federated DataOp is based on input that are also federated.
			// This means that the actual federated leaf nodes will never be reached.
			currentHop.setFederatedOutput(FederatedOutput.FOUT);
		}
		if ( ( isFedInstSupportedHop(currentHop) ) ){
			// The Hop can be FOUT or LOUT or None. Check utility of FOUT vs LOUT vs None.
			currentHop.setFederatedOutput(getHighestUtilFedOut(currentHop));
		}
		else
			currentHop.setFederatedOutput(FEDInstruction.FederatedOutput.NONE);
		currentHop.setVisited();
	}

	/**
	 * Returns the FederatedOutput with the highest utility out of the valid FederatedOutput values.
	 * @param hop for which the utility is found
	 * @return the FederatedOutput value with highest utility for the given Hop
	 */
	private FederatedOutput getHighestUtilFedOut(Hop hop){
		Map<FederatedOutput,Long> fedOutUtilMap = new EnumMap<>(FederatedOutput.class);
		if ( isFOUTSupported(hop) )
			fedOutUtilMap.put(FederatedOutput.FOUT, getUtilFout());
		if ( hop.getPrivacy() == null || (hop.getPrivacy() != null && !hop.getPrivacy().hasConstraints()) )
			fedOutUtilMap.put(FederatedOutput.LOUT, getUtilLout(hop));
		fedOutUtilMap.put(FederatedOutput.NONE, 0L);

		Map.Entry<FederatedOutput, Long> fedOutMax = Collections.max(fedOutUtilMap.entrySet(), Map.Entry.comparingByValue());
		return fedOutMax.getKey();
	}

	/**
	 * Utility if hop is FOUT. This is a simple version where it always returns 1.
	 * @return utility if hop is FOUT
	 */
	private long getUtilFout(){
		//TODO: Make better utility estimation
		return 1;
	}

	/**
	 * Utility if hop is LOUT. This is a simple version only based on dimensions.
	 * @param hop for which utility is calculated
	 * @return utility if hop is LOUT
	 */
	private long getUtilLout(Hop hop){
		//TODO: Make better utility estimation
		return -(long)hop.getMemEstimate();
	}

	private boolean isFedInstSupportedHop(Hop hop){

		// Check that some input is FOUT, otherwise none of the fed instructions will run unless it is fedinit
		if ( (!isFederatedDataOp(hop)) && hop.getInput().stream().noneMatch(Hop::isFederatedOutput) )
			return false;

		// The following operations are supported given that the above conditions have not returned already
		return ( hop instanceof AggBinaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp
			|| hop instanceof AggUnaryOp || hop instanceof TernaryOp || hop instanceof DataOp );
	}

	/**
	 * Checks to see if the associatedHop supports FOUT.
	 * @param associatedHop for which FOUT support is checked
	 * @return true if FOUT is supported by the associatedHop
	 */
	private boolean isFOUTSupported(Hop associatedHop){
		// If the output of AggUnaryOp is a scalar, the operation cannot be FOUT
		if ( associatedHop instanceof AggUnaryOp )
			return !associatedHop.isScalar();
		return true;
	}

	private void visitHop(Hop hop){
		if (hop.isVisited())
			return;

		// Depth first to get to the input
		for ( Hop input : hop.getInput() )
			visitHop(input);

		privacyBasedHopDecisionWithFedCall(hop);
		hop.setVisited();
	}

	private static void privacyBasedHopDecision(Hop hop){
		PrivacyPropagator.hopPropagation(hop);
		PrivacyConstraint privacyConstraint = hop.getPrivacy();
		if ( privacyConstraint != null && privacyConstraint.hasConstraints() )
			hop.setFederatedOutput(FEDInstruction.FederatedOutput.FOUT);
		else if ( hop.someInputFederated() )
			hop.setFederatedOutput(FEDInstruction.FederatedOutput.LOUT);
	}

	/**
	 * Get privacy constraints of DataOps from federated worker,
	 * propagate privacy constraints from input to current hop,
	 * and set federated output flag.
	 * @param hop current hop
	 */
	private static void privacyBasedHopDecisionWithFedCall(Hop hop){
		loadFederatedPrivacyConstraints(hop);
		PrivacyPropagator.hopPropagation(hop);
	}

	/**
	 * Get privacy constraints from federated workers for DataOps.
	 * @hop hop for which privacy constraints are loaded
	 */
	private static void loadFederatedPrivacyConstraints(Hop hop){
		if ( isFederatedDataOp(hop) && hop.getPrivacy() == null){
			try {
				PrivacyConstraint privConstraint = unwrapPrivConstraint(sendPrivConstraintRequest(hop));
				hop.setPrivacy(privConstraint);
			}
			catch(Exception e) {
				throw new DMLException(e.getMessage());
			}
		}
	}

	private static Future<FederatedResponse> sendPrivConstraintRequest(Hop hop)
		throws UnknownHostException, SSLException
	{
		String address = ((LiteralOp) hop.getInput(0).getInput(0)).getStringValue();
		String[] parsedAddress = InitFEDInstruction.parseURL(address);
		String host = parsedAddress[0];
		int port = Integer.parseInt(parsedAddress[1]);
		PrivacyConstraintRetriever retriever = new PrivacyConstraintRetriever(parsedAddress[2]);
		FederatedRequest privacyRetrieval =
			new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1, retriever);
		InetSocketAddress inetAddress = new InetSocketAddress(InetAddress.getByName(host), port);
		return FederatedData.executeFederatedOperation(inetAddress, privacyRetrieval);
	}

	private static PrivacyConstraint unwrapPrivConstraint(Future<FederatedResponse> privConstraintFuture)
		throws Exception
	{
		FederatedResponse privConstraintResponse = privConstraintFuture.get();
		return (PrivacyConstraint) privConstraintResponse.getData()[0];
	}

	private static boolean isFederatedDataOp(Hop hop){
		return hop instanceof DataOp && ((DataOp) hop).isFederatedData();
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
