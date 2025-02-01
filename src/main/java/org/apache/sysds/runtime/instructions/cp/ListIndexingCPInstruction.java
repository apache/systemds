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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Opcodes;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;

public final class ListIndexingCPInstruction extends IndexingCPInstruction {

	protected ListIndexingCPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	protected ListIndexingCPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		ScalarObject rl = ec.getScalarInput(rowLower);
		ScalarObject ru = ec.getScalarInput(rowUpper);
		
		//right indexing
		if( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) ) {
			ListObject list = (ListObject) ec.getVariable(input1.getName());
			
			//execute right indexing operation and set output
			if( rl.getValueType()==ValueType.STRING || ru.getValueType()==ValueType.STRING ) {
				ec.setVariable(output.getName(),
					list.slice(rl.getStringValue(), ru.getStringValue()));
			}
			else {
				ec.setVariable(output.getName(),
					list.slice((int)rl.getLongValue()-1, (int)ru.getLongValue()-1));
			}
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString())) {
			ListObject lin = (ListObject) ec.getVariable(input1.getName());
			
			//execute right indexing operation and set output
			if( input2.getDataType().isList() ) { //LIST <- LIST
				//TODO: copy the lineage trace of input2 list to input1 list
				ListObject rin = (ListObject) ec.getVariable(input2.getName());
				if( rl.getValueType()==ValueType.STRING || ru.getValueType()==ValueType.STRING  )
					ec.setVariable(output.getName(), lin.copy().set(rl.getStringValue(), ru.getStringValue(), rin));
				else
					ec.setVariable(output.getName(), lin.copy().set((int)rl.getLongValue()-1, (int)ru.getLongValue()-1, rin));
			}
			else if( input2.getDataType().isScalar() ) { //LIST <- SCALAR
				ScalarObject scalar = ec.getScalarInput(input2);
				//LineageItem li = DMLScript.LINEAGE ? LineageItemUtils.getLineage(ec, input2)[0] : null; 
				LineageItem li = DMLScript.LINEAGE ? ec.getLineage().getOrCreate(input2) : null; 
				if( rl.getValueType()==ValueType.STRING )
					ec.setVariable(output.getName(), lin.copy().set(rl.getStringValue(), scalar, li));
				else
					ec.setVariable(output.getName(), lin.copy().set((int)rl.getLongValue()-1, scalar, li));
			}
			else if( input2.getDataType().isMatrix() || input2.getDataType().isFrame()) { //LIST <- MATRIX/FRAME
				CacheableData<?> dat = ec.getCacheableData(input2);
				dat.enableCleanup(false);
				LineageItem li = DMLScript.LINEAGE ? ec.getLineage().get(input2) : null;
				if( rl.getValueType()==ValueType.STRING )
					ec.setVariable(output.getName(), lin.copy().set(rl.getStringValue(), dat, li));
				else
					ec.setVariable(output.getName(), lin.copy().set((int)rl.getLongValue()-1, dat, li));
			}
			else {
				throw new DMLRuntimeException("Unsupported list "
					+ "left indexing rhs type: "+input2.getDataType().name());
			}
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in ListIndexingCPInstruction.");
	}
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, input1,input2,input3,rowLower,rowUpper)));
	}
}
