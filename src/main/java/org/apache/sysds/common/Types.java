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

package org.apache.sysds.common;

import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;

/**
 * Common type information for the entire SystemDS.
 */
public interface Types {
	
	/**
	 * Execution mode for entire script. This setting specify which {@link ExecType}s are allowed.
	 */
	public enum ExecMode {
		/** Execute all operations in {@link ExecType#CP} and if available {@link ExecType#GPU} */
		SINGLE_NODE,
		/**
		 * The default and encouraged ExecMode. Execute operations while leveraging all available options:
		 * {@link ExecType#CP}, distributed in {@link ExecType#SPARK}, and if available {@link ExecType#GPU}
		 */
		HYBRID,
		/** Execute all operations in {@link ExecType#SPARK} */
		SPARK
	}
	
	/** Execution type of individual operations. */
	public enum ExecType {
		/** Control Program: This ExecType indicate that the operation should be executed in the controller. */
		CP,
		/**
		 * Control Program File: This ExecType indicate that the operation should be executed in the controller, and the
		 * result be spilled to disk because the intermediate is potentially bigger than memory.
		 */
		CP_FILE,
		/** Spark ExecType indicate that the execution should be performed as a Spark Instruction */
		SPARK,
		/** GPU ExecType indicate that the execution should be performed on a GPU */
		GPU,
		/** FED: indicate that the instruction should be executed as a Federated instruction */
		FED,
		/** invalid is used for debugging or if it is undecided where the current instruction should be executed */
		INVALID
	}

	/** Data types that can contain different ValueTypes internally. */
	public enum DataType{
		/** N Dimensional numeric DataType */
		TENSOR, 
		/** Two or One Dimensional numeric DataType. We use internally Matrix for both Vectors and Matrices */
		MATRIX, 
		/** One cell numeric or String based DataType */
		SCALAR, 
		/** Two or One Dimensional heterogeneous DataType */
		FRAME,
		/** List primitive able to contain any other DataType at indexes*/
		LIST, 
		/** Unknown DataType, used to indicate uncertainty, and for testing */
		UNKNOWN,
		//TODO remove from Data Type -> generic object
		ENCRYPTED_CIPHER, ENCRYPTED_PLAIN;
		
		public boolean isMatrix() {
			return this == MATRIX;
		}

		public boolean isTensor() {
			return this == TENSOR;
		}

		public boolean isFrame() {
			return this == FRAME;
		}

		public boolean isMatrixOrFrame() {
			return isMatrix() || isFrame();
		}

		public boolean isScalar() {
			return this == SCALAR;
		}

		public boolean isList() {
			return this == LIST;
		}

		public boolean isUnknown() {
			return this == UNKNOWN;
		}
	}

	/**
	 * Value types (int, double, string, boolean, unknown).
	 */
	public enum ValueType {
		UINT4, UINT8, // Used for parsing in UINT values from numpy.
		FP32, FP64, INT32, INT64, BOOLEAN, STRING, UNKNOWN,
		HASH64, HASH32, // Indicate that the value is a hash.
		CHARACTER;

		/**
		 * Get if the ValueType is a numeric value.
		 * 
		 * @return If it is numeric.
		 */
		public boolean isNumeric() {
			switch(this) {
				case FP32:
				case FP64:
				case UINT4:
				case UINT8:
				case INT32:
				case INT64:
					return true;
				default:
					return false;
			}
		}
		
		public boolean isFP() {
			return this==FP64 || this==FP32;
		}
		
		/**
		 * Helper method to detect Unknown ValueTypes.
		 * 
		 * @return If the valuetype is Unknown
		 */
		public boolean isUnknown() {
			return this == UNKNOWN;
		}

		/**
		 * If the value is pseudoNumeric, meaning that we can translate it to a number
		 * 
		 * @return If the number is indirectly numeric
		 */
		public boolean isPseudoNumeric() {
			switch(this) {
				case FP32: // Normal numeric
				case FP64:
				case UINT4:
				case UINT8:
				case INT32:
				case INT64:
				case BOOLEAN: // Pseudo Numeric
				case HASH32:
				case HASH64:
				case CHARACTER:
					return true;
				default:
					return false;
			}
		}

		/**
		 * Convert the internal Enum to an externalizable String, the string indicate how the toString version of a value
		 * type acts.
		 * 
		 * @return A capitalized string of the ValueType.
		 */
		public String toExternalString() {
			switch(this) {
				case FP32:
				case FP64:    return "DOUBLE";
				case UINT4:
				case UINT8:
				case INT32:
				case INT64:   return "INT";
				case BOOLEAN: return "BOOLEAN";
				default:      return toString();
			}
		}

		/**
		 * Parse an external string representation to the internal Enum ValueTypes supported.
		 * 
		 * @param value The String to parse
		 * @return The internal Enum of the given string
		 * @throws DMLRuntimeException In case the given string is unsupported or invalid.
		 */
		public static ValueType fromExternalString(String value) {
			//for now we support both internal and external strings
			//until we have completely changed the external types
			if(value == null)
				throw new DMLRuntimeException("Unknown null value type");
			final String lValue = value.toUpperCase();
			switch(lValue) {
				case "FLOAT":
				case "FP32":      return FP32;
				case "FP64":
				case "DOUBLE":    return FP64;
				case "UINT4":     return UINT4;
				case "UINT8":     return UINT8;
				case "INT32":     return INT32;
				case "INT64":
				case "INT":       return INT64;
				case "BOOL":
				case "BOOLEAN":   return BOOLEAN;
				case "STR":
				case "STRING":    return STRING;
				case "CHAR":
				case "CHARACTER": return CHARACTER;
				case "UNKNOWN":   return UNKNOWN;
				case "HASH64":    return HASH64;
				case "HASH32":    return HASH32;
				default:
					throw new DMLRuntimeException("Unknown value type: "+value);
			}
		}

		/**
		 * Given two ValueTypes, would they both print and eternalize similarly.
		 * 
		 * @param vt1 The first ValueType to compare
		 * @param vt2 The second ValueType to compare
		 * @return If they behave similarly.
		 */
		public static boolean isSameTypeString(ValueType vt1, ValueType vt2) {
			return vt1.toExternalString().equals(vt2.toExternalString());
		}

		/**
		 * Get the highest common type, where if one inout is UNKNOWN return the other.
		 * <p>
		 * 
		 * For instance:<p>
		 * Character and String returns String<p>
		 * INT32 and String returns String<p>
		 * INT32 and FP32 returns FP32<p>
		 * INT32 and INT64 returns INT64<p>
		 *
		 * @param a First value type to analyze
		 * @param b Second Value type to analyze
		 * @return The highest common value
		 */
		public static ValueType getHighestCommonTypeSafe(ValueType a, ValueType b) {
			if(a == b)
				return a;
			else
				return getHighestCommonTypeSwitch(a, b);
		}

		/**
		 * Get the highest common type that both ValueTypes can be contained in.
		 * <p>
		 * 
		 * For instance:<p>
		 * Character and String returns String<p>
		 * INT32 and String returns String<p>
		 * INT32 and FP32 returns FP32<p>
		 * INT32 and INT64 returns INT64<p>
		 * 
		 * @param a First ValueType
		 * @param b Second ValueType
		 * @return The common highest type to represent both
		 * @throws DMLRuntimeException If any input is UNKNOWN.
		 */
		public static ValueType getHighestCommonType(ValueType a, ValueType b) {
			if(a == b)
				return a;
			else if(a == UNKNOWN || b == UNKNOWN)
				throw new DMLRuntimeException(
					String.format("Invalid or not implemented support for comparing valueType of: %s and %s", a, b));
			else
				return getHighestCommonTypeSwitch(a, b);
		}

		private static ValueType getHighestCommonTypeSwitch(ValueType a, ValueType b){
			switch(a) {
				case CHARACTER:
					switch(b){
						case UNKNOWN:
							return a;
						default:
							return STRING;
					}
				case HASH32:
					switch(b) {
						case UNKNOWN:
							return a;
						case CHARACTER:
							return STRING;
						case HASH64:
						case STRING:
							return b;
						default:
							return a;
					}
				case HASH64:
					switch(b) {
						case UNKNOWN:
							return a;
						case CHARACTER:
							return STRING;
						case STRING:
							return b;
						default:
							return a;
					}
				case STRING:
					switch(b){
						case UNKNOWN:
							return a;
						default:
							return STRING;
					}
				case FP64:
					switch(b) {
						case HASH64:
						case HASH32:
						case CHARACTER:
							return STRING;
						case UNKNOWN:
							return a;
						case STRING:
							return b;
						default:
							return a;
					}
				case FP32:
					switch(b) {
						case HASH64:
						case HASH32:
						case CHARACTER:
							return STRING;
						case STRING:
						case FP64:
							return b;
						case UNKNOWN:
						default:
							return a;
					}
				case INT64:
					switch(b) {
						case HASH64:
						case HASH32:
							return b;
						case CHARACTER:
							return STRING;
						case STRING:
						case FP64:
						case FP32:
							return b;
						case UNKNOWN:
						default:
							return a;
					}
				case INT32:
					switch(b) {
						case HASH64:
						case HASH32:
							return b;
						case CHARACTER:
							return STRING;
						case STRING:
						case FP64:
						case FP32:
						case INT64:
							return b;
						case UNKNOWN:
						default:
							return a;
					}
				case UINT4:
					switch(b) {
						case HASH64:
						case HASH32:
							return b;
						case CHARACTER:
							return STRING;
						case STRING:
						case FP64:
						case FP32:
						case INT64:
						case INT32:
						case UINT8:
							return b;
						case UNKNOWN:
						default:
							return a;
					}
				case UINT8:
					switch(b) {
						case HASH64:
						case HASH32:
							return b;
						case CHARACTER:
							return STRING;
						case STRING:
						case FP64:
						case FP32:
						case INT64:
						case INT32:
							return b;
						case UNKNOWN:
						default:
							return a;
					}
				case BOOLEAN:
					switch(b){
						case UNKNOWN:
							return a;
						case CHARACTER:
							return STRING;
						default:
							return b;// always higher type in b;
					}
				case UNKNOWN:
				default:
					return b;
			}
		}
	}
	
	/**
	 * Serialization block types (empty, dense, sparse, ultra-sparse)
	 */
	public enum BlockType{
		EMPTY_BLOCK,
		ULTRA_SPARSE_BLOCK,
		SPARSE_BLOCK,
		DENSE_BLOCK,
		DEDUP_BLOCK,
	}
	
	/**
	 * Type of builtin or user-defined function with regard to its
	 * number of return variables.
	 */
	public enum ReturnType {
		NO_RETURN,
		SINGLE_RETURN,
		MULTI_RETURN
	}
	
	
	/**
	 * Type of aggregation direction
	 */
	public enum Direction {
		RowCol, // full aggregate
		Row,    // row aggregate (e.g., rowSums)
		Col;    // column aggregate (e.g., colSums)
		public boolean isRow() {
			return this == Row;
		}
		public boolean isCol() {
			return this == Col;
		}
		public boolean isRowCol() {
			return this == RowCol;
		}
		@Override
		public String toString() {
			switch(this) {
				case RowCol: return "RC";
				case Row:    return "R";
				case Col:    return "C";
				default:
					throw new RuntimeException("Invalid direction type: " + this);
			}
		}
	}

	/** Correction location when performing operations leveraging correcting rounding */
	public enum CorrectionLocationType { 
		NONE, 
		LASTROW, 
		LASTCOLUMN, 
		LASTTWOROWS, 
		LASTTWOCOLUMNS,
		LASTFOURROWS,
		LASTFOURCOLUMNS,
		INVALID;
		
		public int getNumRemovedRowsColumns() {
			return (this==LASTROW || this==LASTCOLUMN) ? 1 :
				(this==LASTTWOROWS || this==LASTTWOCOLUMNS) ? 2 :
				(this==LASTFOURROWS || this==LASTFOURCOLUMNS) ? 4 : 0;
		}
		
		public boolean isRows() {
			return this == LASTROW || this == LASTTWOROWS || this == LASTFOURROWS;
		}
	}
	
	/** Aggregation operations
	 * <p>
	 * The AggOp contain identifying integer values that need to match with their native counterparts for instance for spoof CUDA ops.
	 */
	public enum AggOp {
		SUM, SUM_SQ, MIN, MAX,
		PROD, SUM_PROD,
		TRACE, MEAN, VAR,
		MAXINDEX, MININDEX,
		COUNT_DISTINCT,
		COUNT_DISTINCT_APPROX,
		UNIQUE;

		@Override
		public String toString() {
			switch(this) {
				case SUM:    return "+";
				case SUM_SQ: return "sq+";
				case PROD:   return "*";
				default:     return name().toLowerCase();
			}
		}
		
		public static AggOp valueOfByOpcode(String opcode) {
			switch(opcode) {
				case "+":    return SUM;
				case "sq+":  return SUM_SQ;
				case "*":    return PROD;
				default:     return valueOf(opcode.toUpperCase());
			}
		}
	}
	
	/** Operations that require 1 operand */
	public enum OpOp1 {
		ABS, ACOS, ASIN, ASSERT, ATAN, BROADCAST,
		CAST_AS_FRAME, CAST_AS_LIST, CAST_AS_MATRIX, CAST_AS_SCALAR,
		CAST_AS_BOOLEAN, CAST_AS_DOUBLE, CAST_AS_INT,
		CEIL, CHOLESKY, COS, COSH, CUMMAX, CUMMIN, CUMPROD, CUMSUM,
		CUMSUMPROD, DET, DETECTSCHEMA, COLNAMES, EIGEN, EXISTS, EXP, FLOOR, INVERSE,
		IQM, ISNA, ISNAN, ISINF, LENGTH, LINEAGE, LOG, NCOL, NOT, NROW,
		MEDIAN, PREFETCH, PRINT, ROUND, SIN, SINH, SIGN, SOFTMAX, SQRT, STOP, _EVICT,
		SVD, TAN, TANH, TYPEOF, TRIGREMOTE, SQRT_MATRIX_JAVA,
		//fused ML-specific operators for performance 
		SPROP, //sample proportion: P * (1 - P)
		SIGMOID, //sigmoid function: 1 / (1 + exp(-X))
		LOG_NZ, //sparse-safe log; ppred(X,0,"!=")*log(X)
		
		COMPRESS, DECOMPRESS,
		LOCAL, // instruction to pull data back from spark forcefully and return a CP matrix.

		//low-level operators //TODO used?
		MULT2, MINUS1_MULT, MINUS_RIGHT, 
		POW2, SUBTRACT_NZ;
		

		public boolean isScalarOutput() {
			return this == CAST_AS_SCALAR
				|| this == DET
				|| this == NROW || this == NCOL
				|| this == LENGTH || this == EXISTS
				|| this == IQM || this == LINEAGE
				|| this == MEDIAN;
		}
		
		@Override
		public String toString() {
			switch(this) {
				case CAST_AS_SCALAR:  return "castdts";
				case CAST_AS_MATRIX:  return "castdtm";
				case CAST_AS_FRAME:   return "castdtf";
				case CAST_AS_LIST:    return "castdtl";
				case CAST_AS_DOUBLE:  return "castvtd";
				case CAST_AS_INT:     return "castvti";
				case CAST_AS_BOOLEAN: return "castvtb";
				case CUMMAX:          return Opcodes.UCUMMAX.toString();
				case CUMMIN:          return Opcodes.UCUMMIN.toString();
				case CUMPROD:         return Opcodes.UCUMM.toString();
				case CUMSUM:          return Opcodes.UCUMKP.toString();
				case CUMSUMPROD:      return Opcodes.UCUMKPM.toString();
				case DET:             return Opcodes.DET.toString();
				case DETECTSCHEMA:    return Opcodes.DETECTSCHEMA.toString();
				case MULT2:           return Opcodes.MULT2.toString();
				case NOT:             return Opcodes.NOT.toString();
				case POW2:            return Opcodes.POW2.toString();
				case TYPEOF:          return Opcodes.TYPEOF.toString();
				default:              return name().toLowerCase();
			}
		}

		//need to be kept consistent with toString
		public static OpOp1 valueOfByOpcode(String opcode) {
			switch(opcode) {
				case "castdts": return CAST_AS_SCALAR;
				case "castdtm": return CAST_AS_MATRIX;
				case "castdtf": return CAST_AS_FRAME;
				case "castdtl": return CAST_AS_LIST;
				case "castvtd": return CAST_AS_DOUBLE;
				case "castvti": return CAST_AS_INT;
				case "castvtb": return CAST_AS_BOOLEAN;
				case "ucummax": return CUMMAX;
				case "ucummin": return CUMMIN;
				case "ucum*":   return CUMPROD;
				case "ucumk+":  return CUMSUM;
				case "ucumk+*": return CUMSUMPROD;
				case "detectSchema":    return DETECTSCHEMA;
				case "*2":      return MULT2;
				case "!":       return NOT;
				case "^2":      return POW2;
				case "typeOf":          return TYPEOF;
				default:        return valueOf(opcode.toUpperCase());
			}
		}
	}

	/** Operations that require 2 operands */
	public enum OpOp2 {
		AND(true), APPLY_SCHEMA(false), BITWAND(true), BITWOR(true), BITWSHIFTL(true), BITWSHIFTR(true),
		BITWXOR(true), CBIND(false), COMPRESS(true), CONCAT(false), COV(false), DIV(true),
		DROP_INVALID_TYPE(false), DROP_INVALID_LENGTH(false), EQUAL(true),
		FRAME_ROW_REPLICATE(true), GREATER(true), GREATEREQUAL(true), INTDIV(true),
		INTERQUANTILE(false), IQM(false), LESS(true),
		LESSEQUAL(true), LOG(true), MAX(true), MEDIAN(false), MIN(true),
		MINUS(true), MODULUS(true), MOMENT(false), MULT(true), NOTEQUAL(true), OR(true),
		PLUS(true), POW(true), PRINT(false), QUANTILE(false), SOLVE(false),
		RBIND(false), VALUE_SWAP(false), XOR(true),
		CAST_AS_FRAME(false), // cast as frame with column names
		//fused ML-specific operators for performance
		MINUS_NZ(false), //sparse-safe minus: X-(mean*ppred(X,0,!=))
		LOG_NZ(false), //sparse-safe log; ppred(X,0,"!=")*log(X,0.5)
		MINUS1_MULT(false), //1-X*Y
		QUANTIZE_COMPRESS(false); //quantization-fused compression

		private final boolean _validOuter;
		
		private OpOp2(boolean outer) {
			_validOuter = outer;
		}
		
		public boolean isValidOuter() {
			return _validOuter;
		}
		
		@Override
		public String toString() {
			switch(this) {
				case PLUS:         return Opcodes.PLUS.toString();
				case MINUS:        return Opcodes.MINUS.toString();
				case MINUS_NZ:     return Opcodes.MINUS_NZ.toString();
				case MINUS1_MULT:  return Opcodes.MINUS1_MULT.toString();
				case MULT:         return Opcodes.MULT.toString();
				case DIV:          return Opcodes.DIV.toString();
				case MODULUS:      return Opcodes.MODULUS.toString();
				case INTDIV:       return Opcodes.INTDIV.toString();
				case LESSEQUAL:    return Opcodes.LESSEQUAL.toString();
				case LESS:         return Opcodes.LESS.toString();
				case GREATEREQUAL: return Opcodes.GREATEREQUAL.toString();
				case GREATER:      return Opcodes.GREATER.toString();
				case EQUAL:        return Opcodes.EQUAL.toString();
				case NOTEQUAL:     return Opcodes.NOTEQUAL.toString();
				case OR:           return Opcodes.OR.toString();
				case AND:          return Opcodes.AND.toString();
				case POW:          return Opcodes.POW.toString();
				case IQM:          return "IQM";
				case MOMENT:       return Opcodes.CM.toString();
				case BITWAND:      return Opcodes.BITWAND.toString();
				case BITWOR:       return Opcodes.BITWOR.toString();
				case BITWXOR:      return Opcodes.BITWXOR.toString();
				case BITWSHIFTL:   return Opcodes.BITWSHIFTL.toString();
				case BITWSHIFTR:   return Opcodes.BITWSHIFTR.toString();
				case DROP_INVALID_TYPE: return Opcodes.DROPINVALIDTYPE.toString();
				case DROP_INVALID_LENGTH: return Opcodes.DROPINVALIDLENGTH.toString();
				case FRAME_ROW_REPLICATE: return Opcodes.FREPLICATE.toString();
				case VALUE_SWAP: return Opcodes.VALUESWAP.toString();
				case APPLY_SCHEMA: return Opcodes.APPLYSCHEMA.toString();
				default:           return name().toLowerCase();
			}
		}
		
		//need to be kept consistent with toString
		public static OpOp2 valueOfByOpcode(String opcode) {
			switch(opcode) {
				case "+":           return PLUS;
				case "-":           return MINUS;
				case "-nz":         return MINUS_NZ;
				case "1-*":         return MINUS1_MULT;
				case "*":           return MULT;
				case "/":           return DIV;
				case "%%":          return MODULUS;
				case "%/%":         return INTDIV;
				case "<=":          return LESSEQUAL;
				case "<":           return LESS;
				case ">=":          return GREATEREQUAL;
				case ">":           return GREATER;
				case "==":          return EQUAL;
				case "!=":          return NOTEQUAL;
				case "||":          return OR;
				case "&&":          return AND;
				case "^":           return POW;
				case "IQM":         return IQM;
				case "cm":          return MOMENT;
				case "bitwAnd":     return BITWAND;
				case "bitwOr":      return BITWOR;
				case "bitwXor":     return BITWXOR;
				case "bitwShiftL":  return BITWSHIFTL;
				case "bitwShiftR":  return BITWSHIFTR;
				case "dropInvalidType": return DROP_INVALID_TYPE;
				case "dropInvalidLength": return DROP_INVALID_LENGTH;
				case "freplicate": return FRAME_ROW_REPLICATE;
				case "valueSwap":   return VALUE_SWAP;
				case "applySchema": return APPLY_SCHEMA;
				default:            return valueOf(opcode.toUpperCase());
			}
		}
	}
	
	/** Operations that require 3 operands */
	public enum OpOp3 {
		QUANTILE, INTERQUANTILE, CTABLE, MOMENT, COV, PLUS_MULT, MINUS_MULT, IFELSE, MAP;
		
		@Override
		public String toString() {
			switch(this) {
				case MOMENT:     return Opcodes.CM.toString();
				case PLUS_MULT:  return Opcodes.PM.toString();
				case MINUS_MULT: return Opcodes.MINUSMULT.toString();
				case MAP:          return Opcodes.MAP.toString();
				default:         return name().toLowerCase();
			}
		}
		
		public static OpOp3 valueOfByOpcode(String opcode) {
			switch(opcode) {
				case "cm": return MOMENT;
				case "+*": return PLUS_MULT;
				case "-*": return MINUS_MULT;
				case "_map": return MAP;
				default:   return valueOf(opcode.toUpperCase());
			}
		}
	}
	
	/** Operations that require 4 operands */
	public enum OpOp4 {
		WSLOSS, //weighted sloss mm
		WSIGMOID, //weighted sigmoid mm
		WDIVMM, //weighted divide mm
		WCEMM, //weighted cross entropy mm
		WUMM; //weighted unary mm
		
		@Override
		public String toString() {
			return name().toLowerCase();
		}
	}
	
	/** Operations that require a variable number of operands*/
	public enum OpOpN {
		PRINTF, CBIND, RBIND, MIN, MAX, PLUS, MULT, EVAL, LIST;
		
		public boolean isCellOp() {
			return this == MIN || this == MAX || this == PLUS || this == MULT;
		}
	}
	
	/** Operations that perform internal reorganization of an allocation */
	public enum ReOrgOp {
		DIAG, //DIAG_V2M and DIAG_M2V could not be distinguished if sizes unknown
		RESHAPE, REV, ROLL, SORT, TRANS;
		
		public boolean preservesValues() {
			return this != DIAG && this != SORT;
		}
		
		@Override
		public String toString() {
			switch(this) {
				case DIAG:    return Opcodes.DIAG.toString();
				case TRANS:   return Opcodes.TRANSPOSE.toString();
				case RESHAPE: return Opcodes.RESHAPE.toString();
				default:      return name().toLowerCase();
			}
		}
		
		public static ReOrgOp valueOfByOpcode(String opcode) {
			switch(opcode) {
				case "rdiag":  return DIAG;
				case "r'":     return TRANS;
				case "rshape": return RESHAPE;
				default:       return valueOf(opcode.toUpperCase());
			}
		}
	}
	
	/** Parameterized operations that require named variable arguments */
	public enum ParamBuiltinOp {
		AUTODIFF, CDF, CONTAINS, INVALID, INVCDF, GROUPEDAGG, RMEMPTY, REPLACE, REXPAND,
		LOWER_TRI, UPPER_TRI,
		TRANSFORMAPPLY, TRANSFORMDECODE, TRANSFORMCOLMAP, TRANSFORMMETA,
		TOKENIZE, TOSTRING, LIST, PARAMSERV
	}
	
	/** Deep Neural Network specific operations */
	public enum OpOpDnn {
		MAX_POOL, MAX_POOL_BACKWARD, AVG_POOL, AVG_POOL_BACKWARD,
		CONV2D, CONV2D_BACKWARD_FILTER, CONV2D_BACKWARD_DATA,
		BIASADD, BIASMULT, BATCH_NORM2D_TEST, CHANNEL_SUMS,
		UPDATE_NESTEROV_X,
		//fused operators
		CONV2D_BIAS_ADD, RELU_MAX_POOL, RELU_MAX_POOL_BACKWARD, RELU_BACKWARD
	}
	
	/** Data generation operations */
	public enum OpOpDG {
		RAND, SEQ, FRAMEINIT, SINIT, SAMPLE, TIME
	}
	
	/** Data specific operations, related to reading and writing to and from memory */
	public enum OpOpData {
		PERSISTENTREAD, PERSISTENTWRITE, 
		TRANSIENTREAD, TRANSIENTWRITE,
		FUNCTIONOUTPUT, 
		SQLREAD, FEDERATED;
		
		public boolean isTransient() {
			return this == TRANSIENTREAD || this == TRANSIENTWRITE;
		}
		public boolean isPersistent() {
			return this == PERSISTENTREAD || this == PERSISTENTWRITE;
		}
		public boolean isWrite() {
			return this == TRANSIENTWRITE || this == PERSISTENTWRITE;
		}
		public boolean isRead() {
			return this == TRANSIENTREAD || this == PERSISTENTREAD;
		}
		
		@Override
		public String toString() {
			switch(this) {
				case PERSISTENTREAD:  return "PRead";
				case PERSISTENTWRITE: return "PWrite";
				case TRANSIENTREAD:   return "TRead";
				case TRANSIENTWRITE:  return "TWrite";
				case FUNCTIONOUTPUT:  return "FunOut";
				case SQLREAD:         return Opcodes.SQL.toString();
				case FEDERATED:       return "Fed";
				default:              return "Invalid";
			}
		}
	}

	/** File formats supported */
	public enum FileFormat {
		TEXT,   // text cell IJV representation (mm w/o header)
		MM,     // text matrix market IJV representation
		CSV,    // text dense representation
		COMPRESSED, // Internal SYSTEMDS compressed format 
		LIBSVM, // text libsvm sparse row representation
		JSONL,  // text nested JSON (Line) representation
		BINARY, // binary block representation (dense/sparse/ultra-sparse)
		FEDERATED, // A federated matrix
		PROTO,  // protocol buffer representation
		HDF5,   // Hierarchical Data Format (HDF)
		COG,   // Cloud-optimized GeoTIFF
		PARQUET, // parquet format for columnar data storage
		UNKNOWN;
		
		public boolean isIJV() {
			return this == TEXT || this == MM;
		}
		
		public boolean isTextFormat() {
			return this != BINARY && this != COMPRESSED;
		}
		
		public static boolean isTextFormat(String fmt) {
			try {
				return valueOf(fmt.toUpperCase()).isTextFormat();
			}
			catch(Exception ex) {
				return false;
			}
		}
		
		public boolean isDelimitedFormat() {
			return this == CSV || this == LIBSVM;
		}
		
		public static boolean isDelimitedFormat(String fmt) {
			try {
				return valueOf(fmt.toUpperCase()).isDelimitedFormat();
			}
			catch(Exception ex) {
				return false;
			}
		}
		
		@Override
		public String toString() {
			return name().toLowerCase();
		}

		public static FileFormat defaultFormat() {
			return TEXT;
		}
		
		public static String defaultFormatString() {
			return defaultFormat().toString();
		}
		
		public static FileFormat safeValueOf(String fmt) {
			try {
				return valueOf(fmt.toUpperCase());
			}
			catch(Exception ex) {
				throw new DMLRuntimeException("Unknown file format: "+fmt
					+ " (valid values: " + Arrays.toString(FileFormat.values())+")");
			}
		}
	}
	
	/** Common type for both function statement blocks and function program blocks **/
	public static interface FunctionBlock {
		public FunctionBlock cloneFunctionBlock();
	} 
}
