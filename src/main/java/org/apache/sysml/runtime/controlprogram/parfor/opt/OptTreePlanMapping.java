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

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;
import java.util.Map;

import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;

/**
 * Helper class for mapping nodes of the internal plan representation to statement blocks and 
 * hops / function call statements of a given DML program.
 *
 */
public class OptTreePlanMapping 
{
	
	protected IDSequence _idSeq;
	protected Map<Long, OptNode> _id_optnode;
    
	public OptTreePlanMapping()
	{
		_idSeq = new IDSequence();
		_id_optnode = new HashMap<Long, OptNode>();
	}
	
	/**
	 * 
	 * @param id
	 * @return
	 */
	public OptNode getOptNode( long id )
	{
		return _id_optnode.get(id);
	}
	
	/**
	 * 
	 * @param id
	 * @return
	 */
	public long getMappedParentID( long id )
	{
		for( OptNode p : _id_optnode.values() )
			if( p.getChilds() != null )
				for( OptNode c2 : p.getChilds() )
					if( id == c2.getID() )
						return p.getID();
		return -1;
	}
	
	/**
	 * 
	 */
	public void clear()
	{
		_id_optnode.clear();
	}
	
}
