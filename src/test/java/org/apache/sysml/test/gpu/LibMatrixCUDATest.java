package org.apache.sysml.test.gpu;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.mlcontext.MatrixMetadata;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.MetaData;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Unit tests for GPU methods
 */
public class LibMatrixCUDATest {

	private final long seed1 = 42;
	private final long seed2 = -1;
	private static GPUContext gCtx;
	private static ExecutionContext ec;

	@BeforeClass
	public void setup() throws Exception{
		DMLScript.USE_ACCELERATOR = true;
		DMLScript.FORCE_ACCELERATOR = true;
		DMLScript.rtplatform = DMLScript.RUNTIME_PLATFORM.SINGLE_NODE;
		gCtx = GPUContextPool.getFromPool();
		ec = ExecutionContextFactory.createContext();
	}

	@Test
	public void test1() throws Exception {
		int min = -100;
		int max = 100;
		int rows = 100;
		int cols = 100;
		boolean sparse = true;
		double sparsity = 0.5;

		String inputName = "in1";

		MatrixObject mobj = new MatrixObject(Expression.ValueType.DOUBLE, inputName + "_91" );
		mobj.setVarName(inputName);
		mobj.setDataType(Expression.DataType.MATRIX);
		//clone meta data because it is updated on copy-on-write, otherwise there
		//is potential for hidden side effects between variables.
		MatrixFormatMetaData metadata = new MatrixFormatMetaData(new MatrixCharacteristics(rows, cols, rows, cols), new OutputInfo()
		mobj.setMetaData(metadata);
		mobj.setUpdateType(MatrixObject.UpdateType.COPY);
		ec.setVariable(inputName, mobj);

		MatrixBlock in1 = new MatrixBlock(rows, cols, sparse);
		in1.allocateSparseRowsBlock();
		Random valueR = new Random(seed1);
		Random nnzR = new Random(seed2);

		SparseBlock blk = in1.getSparseBlock();
		int range = max - min;
		for (int i=0; i<rows; i++) {
			for (int j = 0; j < cols; j++) {
				double nnz = nnzR.nextGaussian();
				if (nnz <= sparsity) {
					double v1 = valueR.nextDouble();
					double v = min + range * v1;
					blk.set(i, j, v);
				}
			}
		}

		MatrixBlock out = new MatrixBlock(rows, cols, true);
		out.allocateSparseRowsBlock();
		SparseBlock outBlk = out.getSparseBlock();
		for (int i=0; i<rows; i++){
			for (int j=0; j<cols; j++){
				FastMath.sin(outBlk.get(i,j);
			}
		}

		LibMatrixCUDA.sin(ec, gCtx, null, in, "testOutput");

	}



}