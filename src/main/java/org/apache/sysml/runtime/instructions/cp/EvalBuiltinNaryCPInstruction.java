package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.DataConverter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Eval built-in function instruction
 * Note: it supports only single matrix[double] output
 */
public class EvalBuiltinNaryCPInstruction extends BuiltinNaryCPInstruction {

	public EvalBuiltinNaryCPInstruction(Operator op, String opcode, String istr, CPOperand output, CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		//1. get the namespace and func
		CPOperand func = inputs[0];
		String funcName = func.getName();
		String namespace = null;
		String[] splits = DMLProgram.splitFunctionKey(funcName);
		if (splits.length == 2) {
			namespace = splits[0];
			funcName = splits[1];
		}
		// bound the inputs to avoiding being deleted after the function call
		CPOperand[] boundInputs = Arrays.copyOfRange(inputs, 1, inputs.length);
		ArrayList<String> boundOutputNames = new ArrayList<>();
		boundOutputNames.add(output.getName());
		ArrayList<String> boundInputNames = new ArrayList<>();
		for (CPOperand input : boundInputs) {
			boundInputNames.add(input.getName());
		}

		//2. copy the created output matrix
		MatrixObject outputMO = new MatrixObject(ec.getMatrixObject(output.getName()));

		//3. call the function
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(namespace, funcName, boundInputs, boundInputNames, boundOutputNames, "eval func");
		fcpi.processInstruction(ec);

		//4. convert the result to matrix
		Data newOutput = ec.getVariable(output);
		if (newOutput instanceof MatrixObject) {
			return;
		}
		MatrixBlock mb = null;
		if (newOutput instanceof ScalarObject) {
			//convert scalar to matrix
			mb = new MatrixBlock(((ScalarObject) newOutput).getDoubleValue());
		} else if (newOutput instanceof FrameObject) {
			//convert frame to matrix
			mb = DataConverter.convertToMatrixBlock(((FrameObject) newOutput).acquireRead());
			ec.cleanupCacheableData((FrameObject) newOutput);
		}
		outputMO.acquireModify(mb);
		outputMO.release();
		ec.setVariable(output.getName(), outputMO);
	}
}
