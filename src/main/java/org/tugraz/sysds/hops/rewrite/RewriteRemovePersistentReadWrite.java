/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.hops.rewrite;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * This rewrite is a custom rewrite for JMLC in order to replace all persistent reads
 * and writes with transient reads and writes from the symbol table.
 * 
 * 
 */
public class RewriteRemovePersistentReadWrite extends HopRewriteRule
{
	private static final Log LOG = LogFactory.getLog(RewriteRemovePersistentReadWrite.class.getName());
	
	private HashSet<String> _inputs = null;
	private HashSet<String> _outputs = null;
	private HashMap<String,MetaData> _inputsMeta = null;
	
	public RewriteRemovePersistentReadWrite( String[] in, String[] out ) {
		this(in, out, null);
	}
	
	public RewriteRemovePersistentReadWrite( String[] in, String[] out, LocalVariableMap vars )
	{
		//store input and output names
		_inputs = new HashSet<>();
		for( String var : in )
			_inputs.add( var );
		_outputs = new HashSet<>();
		for( String var : out )
			_outputs.add( var );
		
		//store input meta data
		_inputsMeta = new HashMap<>();
		if( vars != null ) {
			for( String varname : in ) {
				Data dat = vars.get(varname);
				if( dat != null && dat instanceof CacheableData<?> )
					_inputsMeta.put(varname, ((CacheableData<?>)dat).getMetaData());
			}
		}
	}
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return null;
		for( Hop h : roots ) 
			rule_RemovePersistentDataOp( h );
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		rule_RemovePersistentDataOp( root );
		return root;
	}
	
	private void rule_RemovePersistentDataOp( Hop hop ) {
		//check mark processed
		if( hop.isVisited() )
			return;
		
		//recursively process childs
		ArrayList<Hop> inputs = hop.getInput();
		for( int i=0; i<inputs.size(); i++ )
			rule_RemovePersistentDataOp( inputs.get(i) );

		//remove cast if unnecessary
		if( hop instanceof DataOp )
		{
			DataOp dop = (DataOp) hop;
			DataOpTypes dotype = dop.getDataOpType();
			
			switch( dotype ) 
			{
				case PERSISTENTREAD:
					if( _inputs.contains(dop.getName()) ) {
						dop.setDataOpType(DataOpTypes.TRANSIENTREAD);
						if (hop.getDataType() == DataType.SCALAR) {
							dop.removeInput("iofilename");
						}
						
						//disable unnecessary reblock of binary block w/ equal block sizes
						if( dop.requiresReblock() && _inputsMeta.containsKey(dop.getName()) 
							&& _inputsMeta.get(dop.getName()) instanceof MetaDataFormat) {
							MetaDataFormat meta = (MetaDataFormat)_inputsMeta.get(dop.getName());
							DataCharacteristics dc = meta.getDataCharacteristics();
							boolean matchingBlksz = dc.getBlocksize() == dop.getBlocksize();
							//binary matrix w/ matching dims and frames do not require reblock
							if( meta.getInputInfo() == InputInfo.BinaryBlockInputInfo 
								&& (matchingBlksz || dop.getDataType() == DataType.FRAME))
							{
								dop.setRequiresReblock(false);
							}
						}
					} 
					else
						LOG.warn("Non-registered persistent read of variable '"+dop.getName()+"' (line "+dop.getBeginLine()+").");
					break;
				case PERSISTENTWRITE:
					if( _outputs.contains(dop.getName()) ) {
						dop.setDataOpType(DataOpTypes.TRANSIENTWRITE);
						dop.setBlocksize(dop.getInput().get(0).getBlocksize());
						if (hop.getDataType() == DataType.SCALAR) {
							dop.removeInput("iofilename");
						}
					}
					else
						LOG.warn("Non-registered persistent write of variable '"+dop.getName()+"' (line "+dop.getBeginLine()+").");
					break;
				default:
					//do nothing
			}
		}
		
		//mark processed
		hop.setVisited();
	}
}
