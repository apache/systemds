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

package org.apache.sysds.runtime.instructions.gpu;

import java.util.ArrayList;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofCUDAOperator;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.utils.GPUStatistics;

import jcuda.Sizeof;

public class SpoofCUDAInstruction extends GPUInstruction {
	private static final Log LOG = LogFactory.getLog(SpoofCUDAInstruction.class.getName());
	
	public static SpoofCUDAOperator.PrecisionProxy proxy = null;
	
	private final SpoofCUDAOperator _op;
	private final CPOperand[] _in;
	public final CPOperand _out;
	
	public static class SinglePrecision extends SpoofCUDAOperator.PrecisionProxy {
		public int exec(ExecutionContext ec, SpoofCUDAOperator op, int opID, long[] in, long[] sides, long[] out,
				ArrayList<ScalarObject> scalarObjects, long grix) {
			op.setScalarPtr(transferScalars(ec, op, Sizeof.FLOAT, scalarObjects));
			long[] _metadata = { opID, grix, in.length, sides.length, out.length, scalarObjects.size() };
			return op.execute_sp(ctx, _metadata, in, sides, out, GPUObject.getPointerAddress(op.getScalarPtr()));
		}
	}
	
	public static class DoublePrecision extends SpoofCUDAOperator.PrecisionProxy {
		public int exec(ExecutionContext ec, SpoofCUDAOperator op, int opID, long[] in, long[] sides, long[] out,
				ArrayList<ScalarObject> scalarObjects, long grix) {
			if(!scalarObjects.isEmpty())
				op.setScalarPtr(transferScalars(ec, op, Sizeof.DOUBLE, scalarObjects));
			long[] _metadata = { opID, grix, in.length, sides.length, out.length, scalarObjects.size() };
			return op.execute_dp(ctx, _metadata, in, sides, out, GPUObject.getPointerAddress(op.getScalarPtr()));
		}
	}
	
	/**
	 * Sets the internal state based on the DMLScript.DATA_TYPE
	 */
	public static void resetFloatingPointPrecision() {
		if(DMLScript.FLOATING_POINT_PRECISION.equalsIgnoreCase("single")) {
			SpoofCUDAInstruction.proxy = new SinglePrecision();
		}
		else if(DMLScript.FLOATING_POINT_PRECISION.equalsIgnoreCase("double")) {
			SpoofCUDAInstruction.proxy = new DoublePrecision();
		}
		else {
			throw new DMLRuntimeException("Unsupported floating point precision: " + DMLScript.FLOATING_POINT_PRECISION);
		}
	}
	
	private SpoofCUDAInstruction(SpoofCUDAOperator op, CPOperand[] in, CPOperand out, String opcode, String istr) {
		super(null, opcode, istr);
		_op = op;
		_in = in;
		_out = out;
		instString = istr;
		instOpcode = opcode;
	}

	public static SpoofCUDAInstruction parseInstruction(String str) {
		if(proxy == null)
			throw new RuntimeException("SpoofCUDA Executor has not been initialized");
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);

		ArrayList<CPOperand> inlist = new ArrayList<>();
//		Integer op_id =  CodegenUtils.getCUDAopID(parts[2].split("\\.")[1]);
		Integer op_id = CodegenUtils.getCUDAopID(parts[2]);
		Class<?> cla = CodegenUtils.getClass(parts[2]);
		SpoofOperator fallback_java_op = CodegenUtils.createInstance(cla);
		SpoofCUDAOperator op = fallback_java_op.createCUDAInstrcution(op_id, proxy);
		String opcode =  parts[0] + "CUDA" + fallback_java_op.getSpoofType();

		for( int i=3; i<parts.length-2; i++ )
			inlist.add(new CPOperand(parts[i]));
		CPOperand out = new CPOperand(parts[parts.length-2]);

		return new SpoofCUDAInstruction(op, inlist.toArray(new CPOperand[0]), out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		GPUStatistics.incrementNoOfExecutedGPUInst();

		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixObject> inputs = new ArrayList<>();
		ArrayList<ScalarObject> scalars = new ArrayList<>();
		for (CPOperand input : _in) {
			if(input.getDataType()== Types.DataType.MATRIX)
				inputs.add(ec.getMatrixInputForGPUInstruction(input.getName(), getExtendedOpcode()));
			else if(input.getDataType()== Types.DataType.SCALAR) {
				//note: even if literal, it might be compiled as scalar placeholder
				scalars.add(ec.getScalarInput(input));
			}
		}

		try {
			// set the output dimensions to the hop node matrix dimensions
			if(_out.getDataType() == Types.DataType.MATRIX) {
				_op.execute(ec, inputs, scalars, _out.getName());
				ec.releaseMatrixOutputForGPUInstruction(_out.getName());
			}
			else if(_out.getDataType() == Types.DataType.SCALAR) {
				ScalarObject out = _op.execute(ec, inputs, scalars);
				ec.setScalarOutput(_out.getName(), out);
			}
			
			_op.releaseScalarGPUMemory(ec);
		}
		catch(Exception ex) {
			LOG.error("SpoofCUDAInstruction: " + _op.getName() + " operator failed to execute. Trying Java fallback.(ToDo)\n");
			
			throw new DMLRuntimeException(ex);
		}
		
		for (CPOperand input : _in)
			if(input.getDataType()== Types.DataType.MATRIX)
				ec.releaseMatrixInputForGPUInstruction(input.getName());
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(_out.getName(), new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, _in)));
	}
}
