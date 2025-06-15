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

import org.apache.sysds.lops.Append;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.fed.AggregateBinaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AggregateTernaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AggregateUnaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.AppendFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.BinaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.CentralMomentFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.CovarianceFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FEDType;
import org.apache.sysds.runtime.instructions.fed.IndexingFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.QuantilePickFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.QuantileSortFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.ReorgFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.TernaryFEDInstruction;
import org.apache.sysds.runtime.instructions.fed.TsmmFEDInstruction;

import java.util.HashMap;

public class FEDInstructionParser extends InstructionParser
{
	public static final HashMap<String, FEDType> String2FEDInstructionType;
	static {
		String2FEDInstructionType = new HashMap<>();
		String2FEDInstructionType.put( "fedinit"  , FEDType.Init );
		String2FEDInstructionType.put( "tsmm"     , FEDType.Tsmm );
		String2FEDInstructionType.put( "ba+*"     , FEDType.AggregateBinary );
		String2FEDInstructionType.put( "tak+*"    , FEDType.AggregateTernary);

		String2FEDInstructionType.put( "uak+"    , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uark+"   , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uack+"   , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uamax"   , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uacmax"  , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uamin"   , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uacmin"  , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uarmin"  , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uasqk+"  , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uarsqk+" , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uacsqk+" , FEDType.AggregateUnary );
		String2FEDInstructionType.put( "uavar"   , FEDType.AggregateUnary);
		String2FEDInstructionType.put( "uarvar"  , FEDType.AggregateUnary);
		String2FEDInstructionType.put( "uacvar"  , FEDType.AggregateUnary);

		// Arithmetic Instruction Opcodes
		String2FEDInstructionType.put( "+" ,  FEDType.Binary );
		String2FEDInstructionType.put( "-" ,  FEDType.Binary );
		String2FEDInstructionType.put( "*" ,  FEDType.Binary );
		String2FEDInstructionType.put( "/" ,  FEDType.Binary );
		String2FEDInstructionType.put( "1-*", FEDType.Binary); //special * case
		String2FEDInstructionType.put( "^2" , FEDType.Binary); //special ^ case
		String2FEDInstructionType.put( "*2" , FEDType.Binary); //special * case
		String2FEDInstructionType.put( "max", FEDType.Binary );
		String2FEDInstructionType.put( "min", FEDType.Binary );
		String2FEDInstructionType.put( "==",  FEDType.Binary);
		String2FEDInstructionType.put( "!=",  FEDType.Binary);
		String2FEDInstructionType.put( "<",   FEDType.Binary);
		String2FEDInstructionType.put( ">",   FEDType.Binary);
		String2FEDInstructionType.put( "<=",  FEDType.Binary);
		String2FEDInstructionType.put( ">=",  FEDType.Binary);

		// Reorg Instruction Opcodes (repositioning of existing values)
		String2FEDInstructionType.put( "r'"     , FEDType.Reorg );
		String2FEDInstructionType.put( "rdiag"  , FEDType.Reorg );
		String2FEDInstructionType.put( "rev"    , FEDType.Reorg );
		String2FEDInstructionType.put( "roll"    , FEDType.Reorg );
		//String2FEDInstructionType.put( "rshape" , FEDType.Reorg ); Not supported by ReorgFEDInstruction parser!
		//String2FEDInstructionType.put( "rsort"  , FEDType.Reorg ); Not supported by ReorgFEDInstruction parser!

		// Ternary Instruction Opcodes
		String2FEDInstructionType.put( "+*" , FEDType.Ternary);
		String2FEDInstructionType.put( "-*" , FEDType.Ternary);

		//central moment, covariance, quantiles (sort/pick)
		String2FEDInstructionType.put( "cm",    FEDType.CentralMoment);
		String2FEDInstructionType.put( "cov",   FEDType.Covariance);
		String2FEDInstructionType.put( "qsort", FEDType.QSort);
		String2FEDInstructionType.put( "qpick", FEDType.QPick);

		String2FEDInstructionType.put(RightIndex.OPCODE, FEDType.MatrixIndexing);
		String2FEDInstructionType.put(LeftIndex.OPCODE, FEDType.MatrixIndexing);

		String2FEDInstructionType.put(Append.OPCODE, FEDType.Append);
	}

	public static FEDInstruction parseSingleInstruction (String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		FEDType fedtype = InstructionUtils.getFEDType(str);
		if ( fedtype == null )
			throw new DMLRuntimeException("Unable derive fedtype for instruction: " + str);
		FEDInstruction cpinst = parseSingleInstruction(fedtype, str);
		if ( cpinst == null )
			throw new DMLRuntimeException("Unable to parse instruction: " + str);
		return cpinst;
	}
	
	public static FEDInstruction parseSingleInstruction ( FEDType fedtype, String str ) {
		if ( str == null || str.isEmpty() )
			return null;
		switch(fedtype) {
			case Init:
				return InitFEDInstruction.parseInstruction(str);
			case AggregateBinary:
				return AggregateBinaryFEDInstruction.parseInstruction(str);
			case AggregateUnary:
				return AggregateUnaryFEDInstruction.parseInstruction(str);
			case Tsmm:
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
