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

import org.apache.sysml.runtime.matrix.operators.Operator;

public abstract class BinaryMRInstructionBase extends MRInstruction 
{
	
	public byte input1, input2;
	
	public BinaryMRInstructionBase(Operator op, byte in1, byte in2, byte out)
	{
		super(op, out);
		input1=in1;
		input2=in2;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input1, input2};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input1, input2, output};
	}

}
