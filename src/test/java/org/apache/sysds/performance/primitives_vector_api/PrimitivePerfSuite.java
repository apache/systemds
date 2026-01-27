package org.apache.sysds.performance.primitives_vector_api;

public class PrimitivePerfSuite {
  public static void main(String[] args) {
    //int len = BenchUtil.argInt(args, "--len", 262_144);
    int len = BenchUtil.argInt(args, "--len", 1_000_000);
    int warmup = BenchUtil.argInt(args, "--warmup", 10_000);
    int iters = BenchUtil.argInt(args, "--iters", 2000);
    String filter = BenchUtil.argStr(args, "--filter", "");

    for (BenchCase bc : BenchCase.values()) {
      if (!filter.isEmpty() && !bc.name.contains(filter)) continue;

      Ctx ctx = new Ctx();
      ctx.len = len;
      bc.setup.accept(ctx);

      // warm scalar
      ctx.resetC(); 
      BenchUtil.warmup(() -> {bc.scalar.accept(ctx); },warmup);
      ctx.resetC();
      double nsScalar = BenchUtil.measure(() -> { bc.scalar.accept(ctx); }, iters);

      // warm vector
      ctx.resetC(); 
      BenchUtil.warmup(() -> {bc.vector.accept(ctx); }, warmup);
      ctx.resetC();
      double nsVector = BenchUtil.measure(() -> {bc.vector.accept(ctx); }, iters);

      // verify once
      ctx.resetC(); bc.scalar.accept(ctx);
      bc.vector.accept(ctx);
      bc.verify.accept(ctx);

      if (bc.outKind == BenchCase.OutKind.SCALAR_DOUBLE) {
        BenchUtil.printScalarDouble(bc.name, nsScalar, nsVector, ctx.scalarRes, ctx.vectorRes, ctx.ok);
      } else {
        BenchUtil.printArrayDiff(bc.name, nsScalar, nsVector, ctx.maxDiff, ctx.ok);
      }
      
    }
  }
}
