package org.apache.sysds.test.component.matrix;

import java.util.concurrent.TimeUnit;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class BenchmarkMatMult {

    private MatrixBlock left;
    private MatrixBlock right;
    private MatrixBlock result;

    @Param({"1024", "2048", "4096", "8192"})
    public int m;

    @Param({"1"})
    public int cd;

    @Param({"0.5", "0.75", "1"})
    public double sparsityLeft;

    @Param({"0.001", "0.01", "0.1", "0.2"})
    public double sparsityRight;

    @Param({"1024", "2048", "4096", "8192"})
    public int n;

    // Executes once per benchmark
    @Setup(Level.Trial)
    public void setup() {
        left = TestUtils.ceil(TestUtils.generateTestMatrixBlock(m, cd, -10, 10, sparsityLeft, 13));
        if (left.isInSparseFormat()) {
            throw new IllegalStateException("Left matrix is sparse but must be dense for m: " + m + ", cd: " + cd + ", sparsity: " + sparsityLeft);
        }

        MatrixBlock rightOriginal = TestUtils.ceil(TestUtils.generateTestMatrixBlock(cd, n, -10, 10, sparsityRight, 14));
        if (rightOriginal.isInSparseFormat()) {
            right = new MatrixBlock();
            right.copy(rightOriginal, false);
            if (right.isEmptyBlock(false)) {
                throw new IllegalStateException("Right is empty after copy");
            }
            if (right.isEmptyBlock() || right.getNonZeros() != rightOriginal.getNonZeros()) {
                throw new IllegalStateException("Right has weird setup after copying");
            }
        } else {
            throw new IllegalStateException("Right is dense originally");
        }

        result = new MatrixBlock(m, n, true);
        result.allocateBlock();
        if (!result.isInSparseFormat()) {
            throw new IllegalStateException("Expected result is not in sparse format");
        }
    }

    @Setup(Level.Iteration)
    public void clearResult() {
        result.reset(m, n, true);
    }

    @Benchmark
    public void benchmarkDenseDenseOutSparse(Blackhole bh) {
//		LibMatrixMult.matrixMultDenseDenseOutSparse(left, right, result, false, 0, m);
        bh.consume(result);
    }

    public static void main(String[] args) throws RunnerException {
        String tag = "vector-api-mult-dense-dense-out-sparse" + System.currentTimeMillis();
        Options opt = new OptionsBuilder()
                .include(BenchmarkMatMult.class.getName())
                .jvmArgs("--add-modules=jdk.incubator.vector")
                .result(tag + ".csv")
                .resultFormat(ResultFormatType.CSV)
                .build();
        new Runner(opt).run();
    }
}
