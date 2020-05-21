package org.apache.sysds.runtime.codegen;

import java.util.ArrayList;

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

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

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

    public double execute(ArrayList<MatrixObject> inputs, ArrayList<ScalarObject> scalarObjects, MatrixObject out_obj,
                               ExecutionContext ec) {
        double ret = 0;
        long out_ptr = 0;

        if(out_obj != null)
            out_ptr = ec.getGPUPointerAddress(out_obj);

        int offset = 1;
        if(cnt instanceof CNodeOuterProduct)
            offset = 2;

        // only dense input preparation for now
        long[] in_ptrs = new long[offset+1];
        for(int i = 0; i < offset; ++i)
            in_ptrs[i] = ec.getGPUPointerAddress(inputs.get(i));

        long[] side_ptrs = new long[inputs.size() - offset];
        for(int i = offset; i < inputs.size(); ++i)
            side_ptrs[i] = ec.getGPUPointerAddress(inputs.get(i));

        if(isSinglePrecision()) {
            float[] scalars = prepInputScalarsFloat(scalarObjects);

            // ToDo: handle float
           ret = execute_f(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
                    in_ptrs, in_ptrs.length, side_ptrs, side_ptrs.length, out_ptr, scalars,
                    scalars.length, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);

        }
        else {
            double[] scalars = prepInputScalars(scalarObjects);

            ret = execute_d(SpoofCompiler.native_contexts.get(SpoofCompiler.GeneratorAPI.CUDA), name.split("\\.")[1],
                    in_ptrs, in_ptrs.length, side_ptrs, side_ptrs.length, out_ptr, scalars,
                    scalars.length, inputs.get(0).getNumRows(), inputs.get(0).getNumColumns(), 0);
        }
        return ret;
    }

    @Override
    public String getSpoofType() {
        String tmp[] = getClass().getName().split("\\.");
            return  tmp[tmp.length-1] + "_" + getSpoofTemplateType() + "_" + name.split("\\.")[1];
    }

    private native float execute_f(long ctx, String name, long[] in_ptr, long num_inputs, long[] side_ptr, long num_sides,
                                   long out_ptr, float[] scalars, long num_scalars, long m, long n, long grix);

    private native double execute_d(long ctx, String name, long[] in_ptr, long num_inputs, long[] side_ptr, long num_sides,
                                    long out_ptr, double[] scalars, long num_scalars, long m, long n, long grix);
}
