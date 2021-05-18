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

import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.propagation.PrivacyPropagator;

import java.util.ArrayList;

public class RewriteFederatedExecution extends HopRewriteRule {
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if ( roots == null )
			return null;
		for ( Hop root : roots )
			visitHop(root);
		return roots;
	}

	private void visitHop(Hop hop){
		if (hop.isVisited())
			return;

		// Depth first to get to the input
		for ( Hop input : hop.getInput() )
			visitHop(input);

		privacyBasedHopDecisionWithFedCall(hop);
		hop.setVisited();
		// See RewriteAlgebraic SimplificationDynamic.java for reference
	}

	private void privacyBasedHopDecision(Hop hop){
		PrivacyPropagator.hopPropagation(hop);
		PrivacyConstraint privacyConstraint = hop.getPrivacy();
		if ( privacyConstraint != null && privacyConstraint.hasConstraints() ){
			hop.setFederatedOutput(FEDInstruction.FederatedOutput.FOUT);
		}
	}

	private void privacyBasedHopDecisionWithFedCall(Hop hop){
		//TODO: Force getPrivacy call to retrieve privacy constraints from federated workers if not already done
		// This should only be called for DataOps since the other ops get the privacy constraints
		// from the propagations.
		if ( hop instanceof DataOp && ( ((DataOp) hop).getName().equals("X") || ((DataOp) hop).getName().equals("Y") ))
			hop.setPrivacy(new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.PrivateAggregation));
		privacyBasedHopDecision(hop);
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		return null;
	}
}
