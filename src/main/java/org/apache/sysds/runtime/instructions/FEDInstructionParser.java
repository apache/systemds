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

package org.apache.sysds.runtime.instructions;

import org.apache.sysds.common.InstructionType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.fed.AggregateBinaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AggregateTernaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AggregateUnaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AppendFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.BinaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.CentralMomentFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.CovarianceFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.IndexingFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.QuantilePickFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.QuantileSortFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.ReorgFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.TernaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.TsmmFEDInstruction;

public class FEDInstructionParser extends InstructionParser
{

	public static FEDInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		InstructionType fedtype = InstructionUtils.getFEDType(str);
		if ( fedtype == null )
			throw new DMLRuntimeException("Unable derive fedtype for instruction: " + str);
		FEDInstruction cpinst = parseSingleInstruction(fedtype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static FEDInstruction parseSingleInstruction ( InstructionType fedtype, String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		switch(fedtype) {
			case Init:
				return InitFEDInstruction.parseInstruction(str);
			case AggregateBinary:
				return AggregateBinaryFEDInstruction.parseInstruction(str);
			case AggregateUnary:
				return AggregateUnaryFEDInstruction.parseInstruction(str);
			case TSMM:
				return TsmmFEDInstruction.parseInstruction(str);
			case Binary:
				return BinaryFEDInstruction.parseInstruction(str);
			case Ternary:
				return TernaryFEDInstruction.parseInstruction(str);
			case Reorg:
				return ReorgFEDInstruction.parseInstruction(str);
			case Append:
				return AppendFEDInstruction.parseInstruction(str);
			case AggregateTernary:
				return AggregateTernaryFEDInstruction.parseInstruction(str);
			case CentralMoment:
				return CentralMomentFEDInstruction.parseInstruction(str);
			case Covariance:
				return CovarianceFEDInstruction.parseInstruction(str);
			case QSort:
				return QuantileSortFEDInstruction.parseInstruction(str, true);
			case QPick:
				return QuantilePickFEDInstruction.parseInstruction(str);
			case MatrixIndexing:
				return IndexingFEDInstruction.parseInstruction(str);
			default:
				throw new DMLRuntimeException("Invalid FEDERATED Instruction Type: " + fedtype );
		}
	}
}
