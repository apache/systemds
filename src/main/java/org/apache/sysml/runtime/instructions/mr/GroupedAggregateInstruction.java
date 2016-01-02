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

package org.apache.sysml.runtime.instructions.mr;

import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.functionobjects.CM;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;


public class GroupedAggregateInstruction extends UnaryMRInstructionBase
{
	private boolean _weights = false;
	private int _ngroups = -1;
	private long _bclen = -1;
	
	public GroupedAggregateInstruction(Operator op, byte in, byte out, boolean weights, int ngroups, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.GroupedAggregate;
		instString = istr;
		
		_weights = weights;
		_ngroups = ngroups;
	}

	public boolean hasWeights() {
		return _weights;
	}
	
	public int getNGroups() {
		return _ngroups;
	}
	
	public void setBclen(long bclen){
		_bclen = bclen;
	}
	
	public long getBclen(){
		return _bclen;
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput,
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		throw new DMLRuntimeException("GroupedAggregateInstruction.processInstruction() should not be called!");
		
	}

	public static GroupedAggregateInstruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		if(parts.length<3)
			throw new DMLRuntimeException("the number of fields of instruction "+str+" is less than 2!");
		byte in, out;
		String opcode = parts[0];
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[parts.length - 3]);
		boolean weights = Boolean.parseBoolean(parts[parts.length-2]);
		int ngroups = Integer.parseInt(parts[parts.length-1]);
		
		if ( !opcode.equalsIgnoreCase("groupedagg") ) {
			throw new DMLRuntimeException("Invalid opcode in GroupedAggregateInstruction: " + opcode);
		}
		
		Operator optr = parseGroupedAggOperator(parts[2], parts[3]);
		return new GroupedAggregateInstruction(optr, in, out, weights, ngroups, str);
	}
	
	public static Operator parseGroupedAggOperator(String fn, String other) throws DMLRuntimeException {
		AggregateOperationTypes op = AggregateOperationTypes.INVALID;
		if ( fn.equalsIgnoreCase("centralmoment") )
			// in case of CM, we also need to pass "order"
			op = CMOperator.getAggOpType(fn, other);
		else 
			op = CMOperator.getAggOpType(fn, null);
	
		switch(op) {
		case SUM:
			return new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.LASTCOLUMN);
			
		case COUNT:
		case MEAN:
		case VARIANCE:
		case CM2:
		case CM3:
		case CM4:
			return new CMOperator(CM.getCMFnObject(op), op);
		case INVALID:
		default:
			throw new DMLRuntimeException("Invalid Aggregate Operation in GroupedAggregateInstruction: " + op);
		}
	}

}
