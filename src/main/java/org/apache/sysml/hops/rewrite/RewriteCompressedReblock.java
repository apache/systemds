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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.Hop.DataOpTypes;

/**
 * Rule: CompressedReblock: If config compressed.linalg is enabled, we
 * inject compression hooks after pread of matrices w/ both dims &gt; 1.
 */
public class RewriteCompressedReblock extends HopRewriteRule
{
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		boolean compress = ConfigurationManager.getDMLConfig()
				.getBooleanValue(DMLConfig.COMPRESSED_LINALG);
		
		//perform compressed reblock rewrite
		if( compress )
			for( Hop h : roots ) 
				rule_CompressedReblock(h);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		//do nothing (ppred will never occur in predicate)
		return root;
	}

	private void rule_CompressedReblock(Hop hop) 
		throws HopsException 
	{
		// Go to the source(s) of the DAG
		for (Hop hi : hop.getInput()) {
			if (!hi.isVisited())
				rule_CompressedReblock(hi);
		}

		if( hop instanceof DataOp 
			&& ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTREAD
			&& hop.getDim1() > 1 && hop.getDim2() > 1 ) 
		{
			hop.setRequiresCompression(true);
		}

		hop.setVisited();
	}
}
