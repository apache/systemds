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

package org.apache.sysds.runtime.privacy;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.lops.MatMultCP;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FEDType;

import java.util.EnumSet;

/**
 * Class with methods for making decisions about execution type based on privacy constraints and federation of input.
 * These decisions are made at three different levels: (1) HOP-level, (2) LOP-level, and (3) runtime-level.
 */
public class FedDecision {

	private static boolean activated = false;
	private static boolean runtimeConversion = false;
	private static EnumSet<FEDType> allowedFEDInstructions = EnumSet.noneOf(FEDType.class);

	/**
	 * Activate federated decisions based on this class.
	 */
	public static void activate(){
		activated = true;
	}

	/**
	 * Deactivate federated decisions based on this class.
	 */
	public static void deactivate(){
		activated = false;
	}

	/**
	 * Activate conversion of federated instructions to CP instructions during runtime.
	 * If this is activated, federated instructions of the FEDType not specified by allowedFEDInstructions are converted.
	 */
	public static void activateRuntimeConversion(){
		runtimeConversion = true;
	}

	/**
	 * Deactivate conversion of federated instructions to CP instructions during runtime.
	 * If this is deactivated, no federated instructions will be converted by this class during runtime.
	 */
	public static void deactivateRuntimeConversion(){
		runtimeConversion = false;
	}

	/**
	 * Sets the federated instructions that are allowed as federated instructions during runtime.
	 * If runtime conversion is activated, the FED instructions that are not specified here will be
	 * converted to CP instructions. FEDInit is always allowed.
	 * @param allowedFEDInsts set of FEDTypes which are allowed to be federated
	 */
	public static void addAllowedFEDInstructions(EnumSet<FEDType> allowedFEDInsts){
		allowedFEDInstructions = allowedFEDInsts;
		runtimeConversion = true;
	}

	// HOPs

	/**
	 * Return execution type based on given hop.
	 * @param hop for which an execution type needs to be returned
	 * @return execution type for hop
	 */
	public static ExecType hopExecType(Hop hop){
		if (!activated){
			return null;
		}
		if ( hop instanceof AggBinaryOp ){
			return aggBinaryOpExecType((AggBinaryOp) hop);
		}
		else if ( hop instanceof DataOp ){
			return dataOpExecType((DataOp) hop);
		} else return null;
	}

	private static ExecType aggBinaryOpExecType(AggBinaryOp aggBinaryOp){
		if (aggBinaryOp.isMatrixMultiply())
			return ExecType.FED;
		else return null;
	}

	private static ExecType dataOpExecType(DataOp dataOp){
		return ExecType.FED;
	}

	// LOPs

	/**
	 * Return execution type based on given lop.
	 * @param lop for which an execution type needs to be returned
	 * @return execution type for lop
	 */
	public static ExecType lopExectype(Lop lop){
		if ( !activated )
			return null;
		if (lop instanceof MatMultCP)
			return ExecType.FED;
		else
			return null;
	}

	// Runtime

	/**
	 * Returns true if the federated instruction should be converted to CP instruction.
	 * This decision is based on the allowed FED instructions and that this class is activated.
	 * @param instruction FED instruction for which a conversion decision is made
	 * @return true if the given FED instruction should be converted to a CP instruction
	 */
	public static boolean convertToCP(FEDInstruction instruction){
		if ( !activated )
			return false;
		if ( !runtimeConversion )
			return !isFedInit(instruction);
		else return shouldConvert(instruction);
	}

	private static boolean isFedInit(FEDInstruction instruction){
		return (instruction.getFEDInstructionType() == FEDInstruction.FEDType.Init);
	}

	private static boolean shouldConvert(FEDInstruction instruction){
		if ( isFedInit(instruction) )
			return false;
		return !allowedFEDInstructions.contains(instruction.getFEDInstructionType());
	}
}
