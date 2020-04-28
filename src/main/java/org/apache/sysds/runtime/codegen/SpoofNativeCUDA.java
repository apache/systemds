package org.apache.sysds.runtime.codegen;

import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;

public class SpoofNativeCUDA extends SpoofOperator {

    private final SpoofCellwise.CellType _type;
    private final SpoofCellwise.AggOp _aggOp;
    private final boolean _sparseSafe;
    private final boolean _containsSeq;
    public final String _name;

    public SpoofNativeCUDA(String name, SpoofCellwise.CellType type, boolean sparseSafe, boolean containsSeq,
                           SpoofCellwise.AggOp aggOp) {
        _type = type;
        _aggOp = aggOp;
        _sparseSafe = sparseSafe;
        _containsSeq = containsSeq;
        _name = name;
    }

    public String getName() {
        return _name;
    }

    @Override
    public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
        return null;
    }

    public void execute(ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects, MatrixObject out_obj,
                               ExecutionContext ec) {

        // only dense input preparation for now
//        SideInput[] b = prepInputMatrices(inputs, 1, true);
        double[] scalars = prepInputScalars(scalarObjects);

        MatrixObject a = inputs.get(0);


        execute_d(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), _name.split("\\.")[2],
                ec.getGPUPointerAddress(inputs.get(0)), 0, ec.getGPUPointerAddress(out_obj), scalars,
                scalars.length, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);

    }

    @Override
    public String getSpoofType() {
        String tmp[] = getClass().getName().split("\\.");
            return  tmp[tmp.length-1] + "." + _name.split("\\.")[2];
    }

    private native boolean execute_d(long ctx, String name, long in_ptr, long side_ptr, long out_ptr, double[] scalars,
            long num_scalars, long m, long n, long grix);
}
