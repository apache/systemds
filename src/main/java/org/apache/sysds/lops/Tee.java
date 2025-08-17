package org.apache.sysds.lops;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class Tee extends Lop {

    public static final String OPCODE = "tee";
    /**
     * Constructor to be invoked by base class.
     *
     * @param input1  lop type
     * @param dt data type of the output
     * @param vt value type of the output
     */
    public Tee(Lop input1, DataType dt, ValueType vt) {
        super(Lop.Type.Tee, dt, vt);
        this.addInput(input1);
        input1.addOutput(this);
        lps.setProperties(inputs, Types.ExecType.OOC);
    }

    @Override
    public String toString() {
        return "Operation = Tee";
    }

    @Override
    public String getInstructions(String input1, String outputs) {
        // This method generates the instruction string: OOC°tee°input°output1°output2...
        String ret = InstructionUtils.concatOperands(
                getExecType().name(), OPCODE,
                getInputs().get(0).prepInputOperand(input1),
                prepOutputOperand(outputs)
        );

        return ret;
    }
}
