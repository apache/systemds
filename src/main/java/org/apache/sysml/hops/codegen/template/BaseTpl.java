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

package org.apache.sysml.hops.codegen.template;

import java.util.ArrayList;
import java.util.LinkedHashMap;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.runtime.matrix.data.Pair;

public abstract class BaseTpl 
{	
	public enum TemplateType {
		CellTpl,
		OuterProductTpl,
		RowAggTpl
	}
	
	private TemplateType _type = null;
	
	protected ArrayList<Hop> _matrixInputs = new ArrayList<Hop>();
	protected Hop _initialHop;
	protected Hop _endHop;
	protected ArrayList<CNodeData> _initialCnodes = new ArrayList<CNodeData>();
	protected ArrayList<Hop> _adddedMatrices = new ArrayList<Hop>();
	protected boolean _endHopReached = false;

	protected LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> _cpplans = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
	
	protected BaseTpl(TemplateType type) {
		_type = type;
	}
	
	public TemplateType getType() {
		return _type;
	}
	
	public abstract boolean openTpl(Hop hop);

	public abstract boolean findTplBoundaries(Hop initialHop, CplanRegister cplanRegister);
	
	public abstract LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> constructTplCplan(boolean compileLiterals) throws DMLException;
}
