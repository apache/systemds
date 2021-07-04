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

package org.apache.sysds.runtime.privacy.propagation;

import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public enum OperatorType {
	Aggregate,
	NonAggregate;

	/**
	 * Returns the operator type of MMChainCPInstruction based on the input data characteristics.
	 * @param inst MMChainCPInstruction for which operator type is returned
	 * @param ec execution context
	 * @return operator type of instruction
	 */
	public static OperatorType getAggregationType(MMChainCPInstruction inst, ExecutionContext ec){
		DataCharacteristics inputDataCharacteristics = ec.getDataCharacteristics(inst.getInputs()[0].getName());
		if ( inputDataCharacteristics.getRows() == 1 && inputDataCharacteristics.getCols() == 1)
			return NonAggregate;
		else return Aggregate;
	}

	/**
	 * Returns the operator type of MMTSJCPInstruction based on the input data characteristics and the MMTSJType.
	 * @param inst MMTSJCPInstruction for which operator type is returned
	 * @param ec execution context
	 * @return operator type of instruction
	 */
	public static OperatorType getAggregationType(MMTSJCPInstruction inst, ExecutionContext ec){
		DataCharacteristics inputDataCharacteristics = ec.getDataCharacteristics(inst.getInputs()[0].getName());
		if ( (inputDataCharacteristics.getRows() == 1 && inst.getMMTSJType() == MMTSJ.MMTSJType.LEFT)
			|| (inputDataCharacteristics.getCols() == 1 && inst.getMMTSJType() != MMTSJ.MMTSJType.LEFT) )
			return OperatorType.NonAggregate;
		else return OperatorType.Aggregate;
	}

	/**
	 * Returns the operator type of AggregateBinaryCPInstruction based on the input data characteristics and the transpose.
	 * @param inst AggregateBinaryCPInstruction for which operator type is returned
	 * @param ec execution context
	 * @return operator type of instruction
	 */
	public static OperatorType getAggregationType(AggregateBinaryCPInstruction inst, ExecutionContext ec){
		DataCharacteristics inputDC = ec.getDataCharacteristics(inst.input1.getName());
		if ((inputDC.getCols() == 1 && !inst.transposeLeft) || (inputDC.getRows() == 1 && inst.transposeLeft) )
			return OperatorType.NonAggregate;
		else return OperatorType.Aggregate;
	}
}
