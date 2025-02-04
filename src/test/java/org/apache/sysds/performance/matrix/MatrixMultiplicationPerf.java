package org.apache.sysds.performance.matrix;

import java.util.Arrays;

import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstMatrix;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.performance.generators.IGeneratePair;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class MatrixMultiplicationPerf extends APerfTest<Object, Pair<MatrixBlock, MatrixBlock>> {

	// parallelization degree
	private final int k;

	public MatrixMultiplicationPerf(int N, IGenerate<Pair<MatrixBlock, MatrixBlock>> gen, int k) {
		super(N, gen);
		this.k = k;
	}

	public void run() throws Exception {
		warmup(() -> mm(k), 10);
		execute(() -> mm(1), "mm SingleThread", N/10);
		if(k != 1) {
			execute(() -> mm(k), "mm MultiThread: " + k);
		}
	}

	private void mm(int k) {
		Pair<MatrixBlock, MatrixBlock> in = gen.take();
		MatrixBlock left = in.getKey();
		MatrixBlock right = in.getValue();
		left.aggregateBinaryOperations(left, right, InstructionUtils.getMatMultOperator(k));
		ret.add(null);
	}

	@Override
	protected String makeResString() {
		return "";
	}

	public static void main(String[] args) throws Exception {

		IGenerate<MatrixBlock> left;
		IGenerate<MatrixBlock> right;
		final int i;
		final int j;
		final int k;
		final double sp1;
		final double sp2;
		if(args.length == 0) {
			i = Integer.parseInt(args[1]);
			j = Integer.parseInt(args[2]);
			k = Integer.parseInt(args[3]);

			sp1 = Double.parseDouble(args[4]);
			sp2 = Double.parseDouble(args[5]);

		}
		else {

			i = Integer.parseInt(args[1]);
			j = Integer.parseInt(args[2]);
			k = Integer.parseInt(args[3]);

			sp1 = Double.parseDouble(args[4]);
			sp2 = Double.parseDouble(args[5]);

		}

		left = new ConstMatrix(i, j, 10, sp1);
		right = new ConstMatrix(j, k, 10, sp2);
		IGenerate<Pair<MatrixBlock, MatrixBlock>> gen = new IGeneratePair<>(left, right);

		// set number of repeats based on expected number of instructions.

		long inst = (long) i * k * j;

		int N = Math.min(100000, (int) Math.max(100L, 50000000000L / inst));

		System.out.println("MM Perf : rep " +N+ " -- " + Arrays.toString(args));

		new MatrixMultiplicationPerf(N, gen, InfrastructureAnalyzer.getLocalParallelism()).run();
	}
}
