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

package org.apache.sysds.hops.codegen.template;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class TemplateBase 
{	
	public enum TemplateType {
		//ordering specifies type preferences
		MAGG,
		OUTER,
		ROW,
		CELL;
		public int getRank() {
			return this.ordinal();
		}
		public boolean isIn(TemplateType... types) {
			return ArrayUtils.contains(types, this);
		}
	}
	
	public enum CloseType {
		CLOSED_VALID,   //no further fusion, valid entry point
		CLOSED_INVALID, //no further fusion, invalid entry point (to be discarded)
		OPEN_VALID,     //further fusion allowed, valid entry point
		OPEN_INVALID;   //further fusion allowed, but invalid entry point
		public boolean isClosed() {
			return (this == CLOSED_VALID || this == CloseType.CLOSED_INVALID);
		}
		public boolean isValid() {
			return (this == CLOSED_VALID || this == OPEN_VALID);
		}
	}

	protected final TemplateType _type;
	protected final CloseType _ctype;
	
	protected TemplateBase(TemplateType type) {
		this(type, CloseType.OPEN_VALID);
	}
	
	protected TemplateBase(TemplateType type, CloseType ctype) {
		_type = type;
		_ctype = ctype;
	}
	
	public TemplateType getType() {
		return _type;
	}
	
	public CloseType getCType() {
		return _ctype;
	}
	
	public boolean isClosed() {
		return _ctype.isClosed();
	}
	
	@Override
	public int hashCode() {
		return UtilFunctions.intHashCode(
			_type.ordinal(), _ctype.ordinal());
	}
	
	@Override
	public boolean equals(Object obj) {
		if( !(obj instanceof TemplateBase) )
			return false;
		TemplateBase that = (TemplateBase)obj;
		return _type == that._type 
			&& _ctype == that._ctype;
	}
	
	/////////////////////////////////////////////
	// Open-Fuse-Merge-Close interface 
	// (for candidate generation and exploration)
	
	/**
	 * Indicates if this template can be opened at the given hop,
	 * where hop represents bottom (first operation on the inputs) 
	 * of the fused operator.
	 * 
	 * @param hop current hop
	 * @return true if template can be opened 
	 */
	public abstract boolean open(Hop hop);
	
	/**
	 * Indicates if the template can be expanded to the given hop
	 * starting from an open template at the input.
	 * 
	 * @param hop current hop
	 * @param input hop with open template of same type
	 * @return true if the current hop can be fused into the operator.
	 */
	public abstract boolean fuse(Hop hop, Hop input);
	
	/**
	 * Indicates if the template at the current hop can be expanded
	 * by merging another template available for one of its other inputs
	 * which is not yet covered by the template of the current hop. 
	 * 
	 * @param hop current hop
	 * @param input direct input of current hop with available template
	 * @return true if the the input hop can be fused into the current hop
	 */
	public abstract boolean merge(Hop hop, Hop input);
	
	/**
	 * Indicates if the template must be closed at the current hop; either
	 * due to final operations (e.g., aggregate) or unsupported operations.
	 * 
	 * @param hop current hop
	 * @return close type (closed invalid, closed valid, open)
	 */
	public abstract CloseType close(Hop hop);
	
	/////////////////////////////////////////////
	// CPlan construction interface
	// (for plan creation of selected candidates)
	
	/**
	 * Constructs a single cplan rooted at the given hop, according 
	 * to the plan given in the memo structure for this particular 
	 * hop and its recursive inputs.  
	 * 
	 * @param hop root of cplan
	 * @param memo memoization table for partial subplans
	 * @param compileLiterals if true compile non-integer literals 
	 * as constants, otherwise variables. note: integer literals are 
	 * always compiled as constants.
	 * @return pair containing hops and code template
	 */
	public abstract Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals);	
}
