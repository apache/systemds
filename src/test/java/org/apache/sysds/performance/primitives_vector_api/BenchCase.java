package org.apache.sysds.performance.primitives_vector_api;
import org.apache.sysds.performance.primitives_vector_api.BenchCase.OutKind;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;

public enum BenchCase {

    // Aggregations

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


    ROWS_MAXS_VECT_MULT(
      "rowMaxsVectMult dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initDenseB();},
      ctx -> ctx.scalarRes = LibSpoofPrimitives.scalarrowMaxsVectMult(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> ctx.vectorRes = LibSpoofPrimitives.rowMaxsVectMult(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> {
        ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;
      }
    ),

    ROWS_MAXS_VECT_MULT_AIX(
      "rowMaxsVectMult_aix dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> {ctx.initDenseA();ctx.initDenseB();ctx.initDenseAInt();},
      ctx -> {ctx.scalarRes = LibSpoofPrimitives.scalarrowMaxsVectMult(ctx.a, ctx.b, ctx.a_int,0,0,ctx.len);
        BenchUtil.blackhole = ctx.scalarRes;
            },
      ctx -> {
        ctx.vectorRes = LibSpoofPrimitives.rowMaxsVectMult(ctx.a, ctx.b, ctx.a_int,0,0,ctx.len);
        BenchUtil.blackhole = ctx.vectorRes;
            },
      ctx -> {
        ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;
      }
    ),
    VECT_MIN(
      "vectMin dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> ctx.initDenseA(),
      ctx -> {ctx.scalarRes = LibSpoofPrimitives.scalarvectMin(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.scalarRes;
             },
      ctx -> {ctx.vectorRes = LibSpoofPrimitives.vectMin(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.vectorRes;},
      ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
    ),

    VECT_MAX(
      "vectMax dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> ctx.initDenseA(),
      ctx -> {ctx.scalarRes = LibSpoofPrimitives.scalarvectMax(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.scalarRes;
             },
      ctx -> {ctx.vectorRes = LibSpoofPrimitives.vectMax(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.vectorRes;},
      ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
    ),
    VECT_COUNTNNZ(
      "vectCountnnz dense",
      OutKind.SCALAR_DOUBLE,
      ctx -> ctx.initDenseA(),
      ctx -> {ctx.scalarRes = LibSpoofPrimitives.scalarvectCountnnz(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.scalarRes;
             },
      ctx -> {ctx.vectorRes = LibSpoofPrimitives.vectCountnnz(ctx.a, 0, ctx.len);
              BenchUtil.blackhole = ctx.vectorRes;},
      ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
    ),

    // Divisions

    VECT_DIV_ADD(
      "vectDivAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval(); ctx.initDenseADiv();},
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, 0, 0, ctx.len),
      ctx -> LibSpoofPrimitives.vectDivAdd(ctx.a, ctx.bval, ctx.cVector, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    VECT_DIV_ADD_2(
      "vectDivAdd2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.bval, ctx.a, ctx.cScalar, 0, 0, ctx.len),
      ctx -> LibSpoofPrimitives.vectDivAdd(ctx.bval, ctx.a, ctx.cVector, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    VECT_DIV_ADD_SPARSE(
      "vectDivAdd sparse",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initDenseAInt(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, ctx.a_int, 0, 0,ctx.len, ctx.len),
      ctx -> LibSpoofPrimitives.vectDivAdd(ctx.a, ctx.bval, ctx.cVector, ctx.a_int, 0, 0,ctx.len, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),


    VECT_DIV_ADD_SPARSE2(
      "vectDivAdd2 sparse",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initDenseAInt(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.bval, ctx.a, ctx.cScalar, ctx.a_int, 0, 0,ctx.len, ctx.len),
      ctx -> LibSpoofPrimitives.vectDivAdd(ctx.bval, ctx.a, ctx.cVector, ctx.a_int, 0, 0,ctx.len, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    VECT_DIV_WRITE(
      "vectDivWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectDivWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectDivWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_DIV_WRITE2(
      "vectDivWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectDivWrite(ctx.bval, ctx.a, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectDivWrite(ctx.bval, ctx.a, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ), 
    VECT_DIV_WRITE3(
      "vectDivWrite3 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval(); ctx.initDenseBDiv();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectDivWrite(ctx.a, ctx.b, 0, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectDivWrite(ctx.a, ctx.b, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    // Comparisons

    VECT_EQUAL_WRITE(
      "vectEqualWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_EQUAL_ADD(
      "vectEqualAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectEqualAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectEqualAdd(ctx.a, ctx.bval,ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_EQUAL_WRITE2(
      "vectEqualWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_NOTEQUAL_ADD(
      "vectNotequalAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectNotequalAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectNotequalAdd(ctx.a, ctx.bval,ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_NOTEQUAL_WRITE(
      "vectNotequalWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectNotequalWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectNotequalWrite(ctx.a, ctx.bval, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_NOTEQUAL_WRITE2(
      "vectNotequalWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA();  ctx.initDenseB(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectNotequalWrite(ctx.a, ctx.b, 0 ,0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectNotequalWrite(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESS_ADD(
      "vectLessAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectLessAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectLessAdd(ctx.a, ctx.bval,ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESS_WRITE(
      "vectLessWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA();  ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectLessWrite(ctx.a, ctx.bval, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectLessWrite(ctx.a, ctx.bval, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESS_WRITE2(
      "vectLessWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initDenseB(); ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectLessWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectLessWrite(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESSEQUAL_ADD(
      "vectLessequalAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectLessequalAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectLessequalAdd(ctx.a, ctx.bval,ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESSEQUAL_WRITE(
      "vectLessequalWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA();  ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectLessequalWrite(ctx.a, ctx.bval, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectLessequalWrite(ctx.a, ctx.bval, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_LESSEQUAL_WRITE2(
      "vectLessequalWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initDenseB();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectLessequalWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectLessequalWrite(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    VECT_GREATER_ADD(
      "vectGreaterAdd dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval();},
      ctx -> LibSpoofPrimitives.scalarvectGreaterAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectGreaterAdd(ctx.a, ctx.bval,ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_GREATER_WRITE(
      "vectGreaterWrite dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA();  ctx.initbval();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectGreaterWrite(ctx.a, ctx.bval, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectGreaterWrite(ctx.a, ctx.bval, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_GREATER_WRITE2(
      "vectGreaterWrite2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseA(); ctx.initDenseB();},
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectGreaterWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectGreaterWrite(ctx.a, ctx.b, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_DIV_ADD_pure(
      "vectDivAddpure dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); ctx.initbval(); ctx.initDenseADiv();},
      ctx -> LibSpoofPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, 0, 0, ctx.len),
      ctx -> LibSpoofPrimitives.pureDivvectDivAdd(ctx.a, ctx.bval, ctx.cVector, 0, 0, ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),

    // vectMult2

    VECT_Mult2_ADD(
      "vectMult2Add dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); },
      ctx -> LibSpoofPrimitives.scalarvectMult2Add(ctx.a, ctx.cScalar,0, 0,ctx.len),
      ctx -> LibSpoofPrimitives.vectMult2Add(ctx.a, ctx.cVector, 0, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_Mult2_WRITE(
      "vectMult2Write dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); },
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectMult2Write(ctx.a, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectMult2Write(ctx.a, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_Mult2_WRITE_DEDICATED(
      "vectMult2Write_dedicated dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); },
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectMult2Write(ctx.a, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectMult2Write_dedicated(ctx.a, 0,ctx.len),
      ctx -> {
        ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
        ctx.ok = ctx.maxDiff <= 1e-9;
      }
    ),
    VECT_Mult2_WRITE_DEDICATED2(
      "vectMult2Write_dedicated2 dense",
      OutKind.ARRAY_DOUBLE,
      ctx -> {ctx.initDenseAandC_mutable(); },
      ctx -> ctx.cScalar = LibSpoofPrimitives.scalarvectMult2Write(ctx.a, 0,ctx.len),
      ctx -> ctx.cVector = LibSpoofPrimitives.vectMult2Write_dedicated_2(ctx.a, 0,ctx.len),
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
  
