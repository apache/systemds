/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

package org.apache.sysds.runtime.instructions.spark;

import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

/**
 * Class to group the different MM <code>SPInstruction</code>s together.
 */
public abstract class AggregateBinarySPInstruction extends BinarySPInstruction {
	protected AggregateBinarySPInstruction(SPType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}
}
