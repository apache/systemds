package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.api.mlcontext.MLContextConversionUtil;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.Operator;

import java.util.ArrayList;

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
        //1. inside the new instruction, get the correspond function statement block
        FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(null, inputs[0].getName());
        //2. inject the inputs variables into local variable map
        LocalVariableMap lvm = ec.getVariables();
        ArrayList<DataIdentifier> inputParams = fpb.getInputParams();
        if (inputParams.size() != inputs.length - 1) {
            throw new DMLRuntimeException(String.format("processInstruction():: function [%s] should provide %d arguments but only %d arguments provided", inputs[0].getName(), fpb.getInputParams().size(), inputs.length - 1));
        }
        for (int i = 0; i < inputParams.size(); i++) {
            CPOperand arg = inputs[i + 1];
            DataIdentifier di = inputParams.get(i);
            Data d;
            if (di.getDataType().isScalar()) {
                d = ec.getScalarInput(arg.getName(), arg.getValueType(), true);
            } else if (di.getDataType().isMatrix()) {
                d = ec.getMatrixObject(arg.getName());
            } else if (di.getDataType().isFrame()) {
                d = ec.getFrameObject(arg.getName());
            } else {
                throw new DMLRuntimeException(String.format("processInstruction():: unknown argument %s", arg.getName()));
            }
            lvm.put(di.getName(), d);
        }
        fpb.execute(ec);
        //3. get the single func output and convert it to a matrix object
        String funcOutputName = fpb.getOutputParams().get(0).getName();
        Data outputData = ec.getVariable(funcOutputName);
        MatrixObject mo;
        if (outputData.getDataType().isScalar()) {
            ScalarObject so = (ScalarObject) outputData;
            //convert scalar to matrix
            mo = MLContextConversionUtil.doubleMatrixToMatrixObject(output.getName(), new double[][]{{so.getDoubleValue()}});
        } else {
            mo = (MatrixObject) outputData;
        }
        ec.getVariables().put(output.getName(), mo);
    }
}
