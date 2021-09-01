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
import org.apache.sysds.common.Types;
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
import java.util.HashMap;
import java.util.List;
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

	private ArrayList<Hop> selectFederatedExecutionPlan(ArrayList<Hop> roots){
		List<FedPlan> fedPlans = generateFedPlans(roots);
		FedPlan selectedPlan = selectFedPlan(fedPlans);
		selectedPlan.updateHops();

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return null;
		visitHop(root);
		return root;
	}

	private FedPlan selectFedPlan(List<FedPlan> fedPlans){
		//TODO: Different selection algorithm (instead of linear)
		FedPlan bestPlan = fedPlans.get(0);
		if ( fedPlans.size() > 1 ){
			for ( FedPlan planUtility : fedPlans )
				if (planUtility.utility > bestPlan.utility)
					bestPlan = planUtility;
		}
		return bestPlan;
	}

	private List<FedPlan> generateFedPlans(List<Hop> roots){
		List<FedPlan> fedPlans = new ArrayList<>();

		// Create two extreme fed plans
		fedPlans.add(new FedPlan(roots, FEDInstruction.FederatedOutput.FOUT));
		fedPlans.add(new FedPlan(roots, FEDInstruction.FederatedOutput.LOUT));

		return fedPlans;
	}

	class FedPlan {
		private final List<FedPlanRoot> fedPlanRoots = new ArrayList<>();
		private long utility;

		public FedPlan (List<Hop> roots){
			Map<Hop,FedPlanNode> hopToNode = new HashMap<>();
			for ( Hop root : roots ){
				fedPlanRoots.add(new FedPlanRoot(root, hopToNode));
			}
		}

		public FedPlan (List<Hop> roots, FEDInstruction.FederatedOutput strategy){
			this(roots);
			setFedOut(strategy);
			setUtility();
		}

		public void setFedOut(FEDInstruction.FederatedOutput fedOut){
			for ( FedPlanRoot fedPlanRoot : fedPlanRoots ){
				fedPlanRoot.setFedOut(fedOut);
			}
		}

		public long setUtility(){
			fedPlanRoots.forEach(root -> utility += root.getUtility());
			return utility;
		}

		public void updateHops(){
			for ( FedPlanRoot fedPlanRoot : fedPlanRoots )
				fedPlanRoot.updateHops();
		}
	}

	class FedPlanRoot extends FedPlanNode {
		public FedPlanRoot(Hop root, Map<Hop,FedPlanNode> hopToNode){
			super(root, hopToNode);
		}
	}

	class FedPlanChild extends FedPlanNode {
		private final List<FedPlanNode> parents = new ArrayList<>();

		public FedPlanChild(Hop currentHop, Map<Hop,FedPlanNode> hopToNode){
			super(currentHop, hopToNode);
		}

		public void addParent(FedPlanNode parent){
			parents.add(parent);
		}
	}

	abstract class FedPlanNode {
		protected final List<FedPlanNode> children = new ArrayList<>();
		protected Hop associatedHop;
		protected FEDInstruction.FederatedOutput federatedOutput;
		protected Map<Hop,FedPlanNode> hopToNode;
		protected boolean fedOutputVisited = false;
		protected boolean utilityVisited = false;
		protected long utility = 0;

		public FedPlanNode(Hop currentHop, Map<Hop,FedPlanNode> hopToNode){
			this.hopToNode = hopToNode;
			addChildren(currentHop);
			associatedHop = currentHop;
		}

		private void addChildren(Hop currentHop){
			for ( Hop child : currentHop.getInput() ){
				FedPlanChild fedPlanChild;
				if ( !(hopToNode.containsKey(child)) ){
					fedPlanChild = new FedPlanChild(child, hopToNode);
					hopToNode.put(currentHop, fedPlanChild);
				}
				else
					fedPlanChild = (FedPlanChild) hopToNode.get(child);
				fedPlanChild.addParent(this);
				children.add(fedPlanChild);
			}
		}

		protected void setFedOut(FEDInstruction.FederatedOutput strategy){
			for ( FedPlanNode child : children ){
				if ( !child.fedOutputVisited )
					child.setFedOut(strategy);
			}

			if ( ( ( strategy.isForcedFederated() ) && isFedInstSupportedHop(this) )
				|| ( strategy.isForcedLocal() && isFedInstSupportedHop(this)
				&& (associatedHop.getPrivacy() == null
				|| ( associatedHop.getPrivacy() != null && !associatedHop.getPrivacy().hasConstraints() ) )) )
				federatedOutput = strategy;
			else if ( strategy.isForcedLocal() && isFedInstSupportedHop(this)
				&& ( associatedHop.getPrivacy() != null && associatedHop.getPrivacy().hasConstraints() ) )
				federatedOutput = FEDInstruction.FederatedOutput.FOUT;
			else
				federatedOutput = FEDInstruction.FederatedOutput.NONE;

			fedOutputVisited = true;
		}

		public void updateHops(){
			associatedHop.setFederatedOutput(federatedOutput);
			for ( FedPlanNode child : children )
				child.updateHops();
		}

		public long getUtility(){
			//Add utility from children
			for ( FedPlanNode child : children ){
				if ( !child.utilityVisited )
					utility += child.getUtility();
			}

			//Add utility from current node
			utility += estimateUtility();

			utilityVisited = true;

			return utility;
		}

		private long estimateUtility(){
			//TODO: Make better utility estimation
			//Possibly make a class for utility estimators.
			//Quick version by adding one if it is FOUT:
			if ( federatedOutput == FEDInstruction.FederatedOutput.FOUT)
				return 1;
			else
				return 0;
		}
	}

	/**
	 * Hops with supporting federated instructions with parsing and processing based on FederatedOutput flags.
	 * @param node to check for supporting fed instructions
	 * @return true if the hop is supported by a federated instruction
	 */
	private boolean isFedInstSupportedHop(FedPlanNode node){
		Hop hop = node.associatedHop;

		// Check that some input is FOUT, otherwise none of the fed instructions will run unless it is fedinit
		if ( (!(node.associatedHop instanceof DataOp && ((DataOp)node.associatedHop).getOp() == Types.OpOpData.FEDERATED) )
			&& node.children.stream().noneMatch(c -> c.federatedOutput == FEDInstruction.FederatedOutput.FOUT) )
			return false;

		// If the output of AggUnaryOp is a scalar, the operation cannot be FOUT
		// TODO: Can still be LOUT, so this check should be moved so that FOUT/LOUT strategy can be checked
		if ( hop instanceof AggUnaryOp )
			return hop.dimsKnown() && hop.getDim1() > 1 && hop.getDim2() > 1;

		// The following operations are supported given that the above conditions have not returned already
		return ( hop instanceof AggBinaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp
			 || hop instanceof TernaryOp || hop instanceof DataOp );
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
