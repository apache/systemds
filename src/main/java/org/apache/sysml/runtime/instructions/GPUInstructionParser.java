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
package org.apache.sysml.runtime.instructions;

import java.util.HashMap;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import org.apache.sysml.runtime.instructions.gpu.AggregateBinaryGPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.ConvolutionGPUInstruction;

public class GPUInstructionParser  extends InstructionParser 
{
	public static final HashMap<String, CPINSTRUCTION_TYPE> String2GPUInstructionType;
	static {
		String2GPUInstructionType = new HashMap<String, CPINSTRUCTION_TYPE>();
		String2GPUInstructionType.put( "conv2d"      , CPINSTRUCTION_TYPE.Convolution);
		String2GPUInstructionType.put( "conv2d_backward_filter"      , CPINSTRUCTION_TYPE.Convolution);
		String2GPUInstructionType.put( "conv2d_backward_data"      , CPINSTRUCTION_TYPE.Convolution);
		String2GPUInstructionType.put( "ba+*"   	, CPINSTRUCTION_TYPE.AggregateBinary);
		
	}
	
	public static CPInstruction parseSingleInstruction (String str ) 
			throws DMLRuntimeException 
		{
			if ( str == null || str.isEmpty() )
				return null;

			CPINSTRUCTION_TYPE cptype = InstructionUtils.getCPType(str); 
			if ( cptype == null ) 
				throw new DMLRuntimeException("Unable derive cptype for instruction: " + str);
			CPInstruction cpinst = parseSingleInstruction(cptype, str);
			if ( cpinst == null )
				throw new DMLRuntimeException("Unable to parse instruction: " + str);
			return cpinst;
		}
	
	public static CPInstruction parseSingleInstruction ( CPINSTRUCTION_TYPE cptype, String str ) 
			throws DMLRuntimeException 
		{
			
			if ( str == null || str.isEmpty() ) 
				return null;
			
			if(cptype == null) {
				throw new DMLRuntimeException("The instruction is not GPU-enabled:" + str);
			}
			
			switch(cptype) 
			{
				case AggregateBinary:
					return AggregateBinaryGPUInstruction.parseInstruction(str);
					
				case Convolution:
					return ConvolutionGPUInstruction.parseInstruction(str);
					
				default: 
					throw new DMLRuntimeException("Invalid GPU Instruction Type: " + cptype );
			}
			
		}
		
}