package org.apache.sysds.performance.primitives_vector_api;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;

public enum BenchCase {
    VECT_SUM(
      "vectSum dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> ctx.initDenseA(),
      ctx -> {ctx.scalarRes = LibSpoofPrimitives.scalarvectSum(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.scalarRes;
             },
      ctx -> {ctx.vectorRes = LibSpoofPrimitives.vectSum(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.vectorRes;},
      ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
    ),
  
    VECT_DIV_ADD(
      "vectDivAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> ctx.initDenseAandC(),
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, 0, 0, ctx.len),
      ctx -> LibSpoofPrimitives.vectDivAdd(ctx.a, ctx.bval, ctx.cVector, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    );
    public enum OutKind { SCALAR_DOUBLE, ARRAY_DOUBLE }
    public final String name;
    public final java.util.function.Consumer<Ctx> setup;
    public final java.util.function.Consumer<Ctx> scalar;
    public final java.util.function.Consumer<Ctx> vector;
    public final java.util.function.Consumer<Ctx> verify;
    public final OutKind outKind;

  
    BenchCase(String name,
              OutKind outKind,
              java.util.function.Consumer<Ctx> setup,
              java.util.function.Consumer<Ctx> scalar,
              java.util.function.Consumer<Ctx> vector,
              java.util.function.Consumer<Ctx> verify) {
      this.name = name; this.outKind = outKind; this.setup = setup; this.scalar = scalar; this.vector = vector; this.verify = verify;
    }
  }
  
