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

import java.util.Map;

import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.hops.ipa.FunctionCallGraph;
import org.apache.sysds.hops.ipa.FunctionCallSizeInfo;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataExpression;

public abstract class AFederatedPlanner {
	
	/**
	 * Selects a federated execution plan for the given program
	 * by setting the forced execution type.
	 * 
	 * @param prog dml program
	 * @param fgraph function call graph
	 * @param fcallSizes function call graph sizes
	 */
	public abstract void rewriteProgram( DMLProgram prog,
		FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes );
	
	
	protected boolean allowsFederated(Hop hop, Map<Long, FType> fedHops) {
		//generically obtain the input FTypes
		FType[] ft = new FType[hop.getInput().size()];
		for( int i=0; i<hop.getInput().size(); i++ )
			ft[i] = fedHops.get(hop.getInput(i).getHopID());
		
		//handle specific operators
		if( hop instanceof AggBinaryOp ) {
			return (ft[0] != null && ft[1] == null)
				|| (ft[0] == null && ft[1] != null)
				|| (ft[0] == FType.COL && ft[1] == FType.ROW);
		}
		else if( hop instanceof BinaryOp && !hop.getDataType().isScalar() ) {
			return (ft[0] != null && ft[1] == null)
				|| (ft[0] == null && ft[1] != null)
				|| (ft[0] != null && ft[0] == ft[1]);
		}
		else if( hop instanceof TernaryOp && !hop.getDataType().isScalar() ) {
			return (ft[0] != null || ft[1] != null || ft[2] != null);
		}
		else if(ft.length==1 && ft[0] != null) {
			return HopRewriteUtils.isReorg(hop, ReOrgOp.TRANS)
				|| HopRewriteUtils.isAggUnaryOp(hop, AggOp.SUM, AggOp.MIN, AggOp.MAX);
		}
		
		return false;
	}

	/**
	 * Get federated output type of given hop.
	 * LOUT is represented with null.
	 * @param hop current operation
	 * @param fedHops map of hop ID mapped to FType
	 * @return federated output FType of hop
	 */
	protected FType getFederatedOut(Hop hop, Map<Long, FType> fedHops) {
		//generically obtain the input FTypes
		FType[] ft = new FType[hop.getInput().size()];
		for( int i=0; i<hop.getInput().size(); i++ )
			ft[i] = fedHops.get(hop.getInput(i).getHopID());
		
		//handle specific operators
		return getFederatedOut(hop, ft);
	}

	/**
	 * Get FType output of given hop with ft input types.
	 * @param hop given operation for which FType output is returned
	 * @param ft array of input FTypes
	 * @return output FType of hop
	 */
	protected FType getFederatedOut(Hop hop, FType[] ft){
		if( hop instanceof AggBinaryOp ) {
			if( ft[0] != null )
				return ft[0] == FType.ROW ? FType.ROW : null;
			else if( ft[0] != null )
				return ft[0] == FType.COL ? FType.COL : null;
		}
		else if( hop instanceof BinaryOp )
			return ft[0] != null ? ft[0] : ft[1];
		else if( hop instanceof TernaryOp )
			return ft[0] != null ? ft[0] : ft[1] != null ? ft[1] : ft[2];
		else if( HopRewriteUtils.isReorg(hop, ReOrgOp.TRANS) )
			return ft[0] == FType.ROW ? FType.COL : FType.COL;

		return null;
	}
	
	protected FType deriveFType(DataOp fedInit) {
		Hop ranges = fedInit.getInput(fedInit.getParameterIndex(DataExpression.FED_RANGES));
		boolean rowPartitioned = true;
		boolean colPartitioned = true;
		for( int i=0; i<ranges.getInput().size()/2; i++ ) { // workers
			Hop beg = ranges.getInput(2*i);
			Hop end = ranges.getInput(2*i+1);
			long rl = HopRewriteUtils.getIntValueSafe(beg.getInput(0));
			long ru = HopRewriteUtils.getIntValueSafe(end.getInput(0));
			long cl = HopRewriteUtils.getIntValueSafe(beg.getInput(1));
			long cu = HopRewriteUtils.getIntValueSafe(end.getInput(1));
			rowPartitioned &= (cu-cl == fedInit.getDim2());
			colPartitioned &= (ru-rl == fedInit.getDim1());
		}
		return rowPartitioned && colPartitioned ?
			FType.FULL : rowPartitioned ? FType.ROW :
			colPartitioned ? FType.COL : FType.OTHER;
	}
}
