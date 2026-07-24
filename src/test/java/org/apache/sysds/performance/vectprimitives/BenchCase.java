/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


package org.apache.sysds.performance.vectprimitives;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;


public enum BenchCase {

	// Aggregations

	VECT_SUM(
	  "vectSum dense",
	  OutKind.SCALAR_DOUBLE,
	  ctx -> ctx.initDenseA(),
	  ctx -> {ctx.scalarRes = BenchmarkPrimitives.scalarvectSum(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.scalarRes;
			 },
	  ctx -> {ctx.vectorRes = BenchmarkPrimitives.vectSum(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.vectorRes;},
	  ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
	),


	ROWS_MAXS_VECT_MULT(
	  "rowMaxsVectMult dense",
	  OutKind.SCALAR_DOUBLE,
	  ctx -> {ctx.initDenseA(); ctx.initDenseB();},
	  ctx -> ctx.scalarRes = BenchmarkPrimitives.scalarrowMaxsVectMult(ctx.a, ctx.b, 0, 0, ctx.len),
	  ctx -> ctx.vectorRes = BenchmarkPrimitives.rowMaxsVectMult(ctx.a, ctx.b, 0, 0, ctx.len),
	  ctx -> {
		ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;
	  }
	),

	ROWS_MAXS_VECT_MULT_AIX(
	  "rowMaxsVectMult_aix dense",
	  OutKind.SCALAR_DOUBLE,
	  ctx -> {ctx.initDenseA();ctx.initDenseB();ctx.initDenseAInt();},
	  ctx -> {ctx.scalarRes = BenchmarkPrimitives.scalarrowMaxsVectMult(ctx.a, ctx.b, ctx.a_int,0,0,ctx.len);
		BenchUtil.blackhole = ctx.scalarRes;
			},
	  ctx -> {
		ctx.vectorRes = BenchmarkPrimitives.rowMaxsVectMult(ctx.a, ctx.b, ctx.a_int,0,0,ctx.len);
		BenchUtil.blackhole = ctx.vectorRes;
			},
	  ctx -> {
		ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;
	  }
	),

	VECT_MAX(
	  "vectMax dense",
	  OutKind.SCALAR_DOUBLE,
	  ctx -> ctx.initDenseA(),
	  ctx -> {ctx.scalarRes = BenchmarkPrimitives.scalarvectMax(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.scalarRes;
			 },
	  ctx -> {ctx.vectorRes = BenchmarkPrimitives.vectMax(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.vectorRes;},
	  ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
	),
	VECT_COUNTNNZ(
	  "vectCountnnz dense",
	  OutKind.SCALAR_DOUBLE,
	  ctx -> ctx.initDenseA(),
	  ctx -> {ctx.scalarRes = BenchmarkPrimitives.scalarvectCountnnz(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.scalarRes;
			 },
	  ctx -> {ctx.vectorRes = BenchmarkPrimitives.vectCountnnz(ctx.a, 0, ctx.len);
			  BenchUtil.blackhole = ctx.vectorRes;},
	  ctx -> {ctx.ok = Math.abs(ctx.scalarRes - ctx.vectorRes) <= 1e-9;}
	),

	// Divisions

	VECT_DIV_ADD(
	  "vectDivAdd dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval(); ctx.initDenseADiv();},
	  ctx -> BenchmarkPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, 0, 0, ctx.len),
	  ctx -> BenchmarkPrimitives.vectDivAdd(ctx.a, ctx.bval, ctx.cVector, 0, 0, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),

	VECT_DIV_ADD_2(
	  "vectDivAdd2 dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectDivAdd(ctx.bval, ctx.a, ctx.cScalar, 0, 0, ctx.len),
	  ctx -> LibSpoofPrimitives.vectDivAdd(ctx.bval, ctx.a, ctx.cVector, 0, 0, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),

	VECT_DIV_ADD_SPARSE(
	  "vectDivAdd sparse",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initDenseAInt(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectDivAdd(ctx.a, ctx.bval, ctx.cScalar, ctx.a_int, 0, 0,ctx.len, ctx.len),
	  ctx -> LibSpoofPrimitives.vectDivAdd(ctx.a, ctx.bval, ctx.cVector, ctx.a_int, 0, 0,ctx.len, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),


	VECT_DIV_ADD_SPARSE2(
	  "vectDivAdd2 sparse",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initDenseAInt(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectDivAdd(ctx.bval, ctx.a, ctx.cScalar, ctx.a_int, 0, 0,ctx.len, ctx.len),
	  ctx -> LibSpoofPrimitives.vectDivAdd(ctx.bval, ctx.a, ctx.cVector, ctx.a_int, 0, 0,ctx.len, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),

	VECT_DIV_WRITE(
	  "vectDivWrite dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectDivWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectDivWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),
	VECT_DIV_WRITE2(
	  "vectDivWrite2 dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectDivWrite(ctx.bval, ctx.a, 0,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectDivWrite(ctx.bval, ctx.a, 0,ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	), 
	VECT_DIV_WRITE3(
	  "vectDivWrite3 dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval(); ctx.initDenseBDiv();},
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectDivWrite(ctx.a, ctx.b, 0, 0,ctx.len),
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
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),
	VECT_EQUAL_ADD(
	  "vectEqualAdd dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectEqualAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectEqualWrite(ctx.a, ctx.bval, 0,ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),
	VECT_LESS_ADD(
	  "vectLessAdd dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectLessAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectLessWrite(ctx.a, ctx.bval, 0 ,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectLessWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectLessWrite(ctx.a, ctx.b, 0, 0, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),
	VECT_LESSEQUAL_ADD(
	  "vectLessequalAdd dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectLessequalAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectLessequalWrite(ctx.a, ctx.bval, 0 ,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectLessequalWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectLessequalWrite(ctx.a, ctx.b, 0, 0, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),

	VECT_GREATER_ADD(
	  "vectGreaterAdd dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); ctx.initbval();},
	  ctx -> BenchmarkPrimitives.scalarvectGreaterAdd(ctx.a, ctx.bval, ctx.cScalar,0, 0,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectGreaterWrite(ctx.a, ctx.bval, 0 ,ctx.len),
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
	  ctx -> ctx.cScalar = BenchmarkPrimitives.scalarvectGreaterWrite(ctx.a, ctx.b, 0, 0 ,ctx.len),
	  ctx -> ctx.cVector = LibSpoofPrimitives.vectGreaterWrite(ctx.a, ctx.b, 0, 0, ctx.len),
	  ctx -> {
		ctx.maxDiff = BenchUtil.maxAbsDiff(ctx.cScalar, ctx.cVector);
		ctx.ok = ctx.maxDiff <= 1e-9;
	  }
	),

	// vectMult2

	VECT_Mult2_ADD(
	  "vectMult2Add dense",
	  OutKind.ARRAY_DOUBLE,
	  ctx -> {ctx.initDenseACMutable(); },
	  ctx -> BenchmarkPrimitives.scalarvectMult2Add(ctx.a, ctx.cScalar,0, 0,ctx.len),
	  ctx -> LibSpoofPrimitives.vectMult2Add(ctx.a, ctx.cVector, 0, 0,ctx.len),
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

