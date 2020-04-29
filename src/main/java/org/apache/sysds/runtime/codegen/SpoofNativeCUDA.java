package org.apache.sysds.runtime.codegen;

import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysds.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;

public class SpoofNativeCUDA extends SpoofOperator {

    private final CNodeTpl cnt;
    public final String name;

    public SpoofNativeCUDA(CNodeTpl cnode) {
        name = "codegen." + cnode.getVarname();
        cnt = cnode;
    }

    public String getName() {
        return name;
    }

    public String getSpoofTemplateType() {
        if (cnt instanceof CNodeCell)
            return "CW";
        else if(cnt instanceof CNodeRow)
            return "RA";
        else if(cnt instanceof CNodeMultiAgg)
            return "MA";
        else if(cnt instanceof CNodeOuterProduct)
            return "OP";
        else
            throw new RuntimeException("unknown spoof operator type");
    }
    @Override
    public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
        throw new RuntimeException("method not implemented for SpoofNativeCUDA");
    }

    public void execute(ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects, MatrixObject out_obj,
                               ExecutionContext ec) {

        // only dense input preparation for now
//        SideInput[] b = prepInputMatrices(inputs, 1, true);
        double[] scalars = prepInputScalars(scalarObjects);

        MatrixObject a = inputs.get(0);


        execute_d(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
                ec.getGPUPointerAddress(inputs.get(0)), 0, ec.getGPUPointerAddress(out_obj), scalars,
                scalars.length, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);

    }

    @Override
    public String getSpoofType() {
        String tmp[] = getClass().getName().split("\\.");
            return  tmp[tmp.length-1] + "_" + getSpoofTemplateType() + "_" + name.split("\\.")[1];
    }

    private native boolean execute_d(long ctx, String name, long in_ptr, long side_ptr, long out_ptr, double[] scalars,
            long num_scalars, long m, long n, long grix);
}
