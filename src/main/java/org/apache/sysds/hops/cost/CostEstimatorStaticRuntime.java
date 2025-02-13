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

package org.apache.sysds.hops.cost;

import org.apache.sysds.common.InstructionType;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class CostEstimatorStaticRuntime extends CostEstimator
{
	//time-conversion
	private static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; //2GFLOPS
	//private static final long UNKNOWN_TIME = -1;
	
	//floating point operations
	private static final double DEFAULT_NFLOP_NOOP = 10; 
	private static final double DEFAULT_NFLOP_UNKNOWN = 1; 
	private static final double DEFAULT_NFLOP_CP = 1; 	
	private static final double DEFAULT_NFLOP_TEXT_IO = 350; 
	
	//IO READ throughput
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE = 200;
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE = 100;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE = 150;
	public static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE = 75;
	//IO WRITE throughput
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE = 120;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE = 60;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_DENSE = 40;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE = 30;
	
	@Override
	protected double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args )
	{
		CPInstruction cpinst = (CPInstruction)inst;
		
		//load time into mem
		double ltime = 0;
		if( !vs[0]._inmem ){
			ltime += getHDFSReadTime( vs[0].getRows(), vs[0].getCols(), vs[0].getSparsity() );
			//eviction costs
			if( LazyWriteBuffer.getWriteBufferLimit()<MatrixBlock.estimateSizeOnDisk(vs[0].getRows(), vs[0].getCols(), 
					(vs[0]._dc.getNonZeros()<0)? vs[0].getRows()*vs[0].getCols():vs[0]._dc.getNonZeros()) )
			{
				ltime += Math.abs( getFSWriteTime( vs[0].getRows(), vs[0].getCols(), vs[0].getSparsity() ));
			}
			vs[0]._inmem = true;
		}
		if( !vs[1]._inmem ){
			ltime += getHDFSReadTime( vs[1].getRows(), vs[1].getCols(), vs[1].getSparsity() );
			//eviction costs
			if( LazyWriteBuffer.getWriteBufferLimit()<MatrixBlock.estimateSizeOnDisk(vs[1].getRows(), vs[1].getCols(), (vs[1]._dc.getNonZeros()<0)? vs[1].getRows()*vs[1].getCols():vs[1]._dc.getNonZeros()) )
			{
				ltime += Math.abs( getFSWriteTime( vs[1].getRows(), vs[1].getCols(), vs[1].getSparsity()) );
			}
			vs[1]._inmem = true;
		}
		if( LOG.isDebugEnabled() && ltime!=0 ) {
			LOG.debug("Cost["+cpinst.getOpcode()+" - read] = "+ltime);
		}		
				
		//exec time CP instruction
		String opcode = (cpinst instanceof FunctionCallCPInstruction) ? InstructionUtils.getOpCode(cpinst.toString()) : cpinst.getOpcode();
		double etime = getInstTimeEstimate(opcode, vs, args, ExecType.CP);
		
		//write time caching
		double wtime = 0;
		//double wtime = getFSWriteTime( vs[2]._rlen, vs[2]._clen, (vs[2]._nnz<0)? 1.0:(double)vs[2]._nnz/vs[2]._rlen/vs[2]._clen );
		if( inst instanceof VariableCPInstruction && ((VariableCPInstruction)inst).getOpcode().equals(Opcodes.WRITE.toString()) )
			wtime += getHDFSWriteTime(vs[2].getRows(), vs[2].getCols(), vs[2].getSparsity(), ((VariableCPInstruction)inst).getInput3().getName() );
		
		if( LOG.isDebugEnabled() && wtime!=0 ) {
			LOG.debug("Cost["+cpinst.getOpcode()+" - write] = "+wtime);
		}
		
		//total costs
		double costs = ltime + etime + wtime;
		
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs CP instruction = "+costs);
		
		return costs;
	}
	
	/////////////////////
	// I/O Costs       //
	/////////////////////	
	
	/**
	 * Returns the estimated read time from HDFS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm rows?
	 * @param dn columns?
	 * @param ds sparsity factor?
	 * @return estimated HDFS read time
	 */
	private static double getHDFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		
		if( sparse )
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}
	
	private static double getHDFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);
		
		double ret = -1;
		if( sparse )
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		
		return ret;
	}
	
	private static double getHDFSWriteTime( long dm, long dn, double ds, String format )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);
		
		double ret = -1;
		
		FileFormat fmt = FileFormat.safeValueOf(format);
		if( fmt.isTextFormat() ) {
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_DENSE;
			ret *= 2.75; //text commonly 2x-3.5x larger than binary
		}
		else {
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		}
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		return ret;
	}

	/**
	 * Returns the estimated read time from local FS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm rows?
	 * @param dn columns?
	 * @param ds sparsity factor?
	 * @return estimated local file system read time
	 */
	public static double getFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		if( sparse )
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}

	public static double getFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		
		if( sparse )
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE;
		
		return ret;
	}

	
	/////////////////////
	// Operation Costs //
	/////////////////////
	
	private static double getInstTimeEstimate(String opcode, VarStats[] vs, String[] args, ExecType et) {
		return getInstTimeEstimate(opcode, false,
			vs[0].getRows(), vs[0].getCols(), !vs[0]._dc.nnzKnown() ? 1.0 : vs[0].getSparsity(),
			vs[1].getRows(), vs[1].getCols(), !vs[1]._dc.nnzKnown() ? 1.0 : vs[1].getSparsity(),
			vs[2].getRows(), vs[2].getCols(), !vs[2]._dc.nnzKnown() ? 1.0 : vs[2].getSparsity(),
			args);
	}
	
	/**
	 * Returns the estimated instruction execution time, w/o data transfer and single-threaded.
	 * For scalars input dims must be set to 1 before invocation. 
	 * 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param opcode instruction opcode
	 * @param inMR ?
	 * @param d1m ?
	 * @param d1n ?
	 * @param d1s ?
	 * @param d2m ?
	 * @param d2n ?
	 * @param d2s ?
	 * @param d3m ?
	 * @param d3n ?
	 * @param d3s ?
	 * @param args ?
	 * @return estimated instruction execution time
	 */
	private static double getInstTimeEstimate( String opcode, boolean inMR, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args )
	{
		double nflops = getNFLOP(opcode, inMR, d1m, d1n, d1s, d2m, d2n, d2s, d3m, d3n, d3s, args);
		double time = nflops / DEFAULT_FLOPS;
		
		if( LOG.isDebugEnabled() )
			LOG.debug("Cost["+opcode+"] = "+time+"s, "+nflops+" flops ("+d1m+","+d1n+","+d1s+","+d2m+","+d2n+","+d2s+","+d3m+","+d3n+","+d3s+").");
		
		return time;
	}
	
	private static double getNFLOP( String optype, boolean inMR, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args )
	{
		//operation costs in FLOP on matrix block level (for CP and MR instructions)
		//(excludes IO and parallelism; assumes known dims for all inputs, outputs )
	
		boolean leftSparse = MatrixBlock.evalSparseFormatInMemory(d1m, d1n, (long)(d1s*d1m*d1n));
		boolean rightSparse = MatrixBlock.evalSparseFormatInMemory(d2m, d2n, (long)(d2s*d2m*d2n));
		boolean onlyLeft = (d1m>=0 && d1n>=0 && d2m<0 && d2n<0 );
		boolean allExists = (d1m>=0 && d1n>=0 && d2m>=0 && d2n>=0 && d3m>=0 && d3n>=0 );
		
		//NOTE: all instruction types that are equivalent in CP and MR are only
		//included in CP to prevent redundancy
		InstructionType cptype = Opcodes.getTypeByOpcode(optype, Types.ExecType.CP);
		if( cptype != null ) //for CP Ops and equivalent MR ops
		{
			//general approach: count of floating point *, /, +, -, ^, builtin ;
			switch(cptype) 
			{
			
				case AggregateBinary: //opcodes: ba+*, cov
					if( optype.equals(Opcodes.MMULT.toString()) ) { //matrix mult
						//reduction by factor 2 because matrix mult better than
						//average flop count
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * ((d2n>1)?d1s:1.0) * d2n) /2;
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n) /2;
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
					}
					else if( optype.equals(Opcodes.COV.toString()) ) {
						//note: output always scalar, d3 used as weights block
						//if( allExists ), same runtime for 2 and 3 inputs
						return 23 * d1m; //(11+3*k+)
					}
					
					return 0;
				
				case MMChain:
					//reduction by factor 2 because matrix mult better than average flop count
					//(mmchain essentially two matrix-vector muliplications)
					if( !leftSparse  )
						return (2+2) * (d1m * d1n) /2;
					else 
						return (2+2) * (d1m * d1n * d1s) /2;
					
				case AggregateTernary: //opcodes: tak+*
					return 6 * d1m * d1n; //2*1(*) + 4 (k+)
					
				case AggregateUnary: //opcodes: uak+, uark+, uack+, uasqk+, uarsqk+, uacsqk+,
				                     //         uamean, uarmean, uacmean, uavar, uarvar, uacvar,
				                     //         uamax, uarmax, uarimax, uacmax, uamin, uarmin, uacmin,
				                     //         ua+, uar+, uac+, ua*, uatrace, uaktrace,
				                     //         nrow, ncol, length, cm
					
					if( optype.equals("nrow") || optype.equals("ncol") || optype.equals("length") )
						return DEFAULT_NFLOP_NOOP;
					else if( optype.equals( Opcodes.CM.toString() ) ) {
						double xcm = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xcm=1; break; //count
							case 1: xcm=8; break; //mean
							case 2: xcm=16; break; //cm2
							case 3: xcm=31; break; //cm3
							case 4: xcm=51; break; //cm4
							case 5: xcm=16; break; //variance
						}
						return (leftSparse) ? xcm * (d1m * d1s + 1) : xcm * d1m;
					}
					else if( optype.equals(Opcodes.UATRACE.toString()) || optype.equals(Opcodes.UAKTRACE.toString()) )
						return 2 * d1m * d1n;
					else if( optype.equals(Opcodes.UAP.toString()) || optype.equals(Opcodes.UARP.toString()) || optype.equals(Opcodes.UACP.toString())  ){
						//sparse safe operations
						if( !leftSparse ) //dense
							return d1m * d1n;
						else //sparse
							return d1m * d1n * d1s;
					}
					else if( optype.equals(Opcodes.UAKP.toString()) || optype.equals(Opcodes.UARKP.toString()) || optype.equals(Opcodes.UACKP.toString()))
						return 4 * d1m * d1n; //1*k+
					else if( optype.equals(Opcodes.UASQKP.toString()) || optype.equals(Opcodes.UARSQKP.toString()) || optype.equals(Opcodes.UACSQKP.toString()))
						return 5 * d1m * d1n; // +1 for multiplication to square term
					else if( optype.equals(Opcodes.UAMEAN.toString()) || optype.equals(Opcodes.UARMEAN.toString()) || optype.equals(Opcodes.UACMEAN.toString()))
						return 7 * d1m * d1n; //1*k+
					else if( optype.equals(Opcodes.UAVAR.toString()) || optype.equals(Opcodes.UARVAR.toString()) || optype.equals(Opcodes.UACVAR.toString()))
						return 14 * d1m * d1n;
					else if(   optype.equals(Opcodes.UAMAX.toString()) || optype.equals(Opcodes.UARMAX.toString()) || optype.equals(Opcodes.UACMAX.toString())
						|| optype.equals(Opcodes.UAMIN.toString()) || optype.equals(Opcodes.UARMIN.toString()) || optype.equals(Opcodes.UACMIN.toString())
						|| optype.equals(Opcodes.UARIMAX.toString()) || optype.equals(Opcodes.UAM.toString()) )
						return d1m * d1n;
					
					return 0;
				
				case Binary: //opcodes: +, -, *, /, ^ (incl. ^2, *2),
					//max, min, solve, ==, !=, <, >, <=, >=  
					//note: all relational ops are not sparsesafe
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					if( optype.equals(Opcodes.PLUS.toString()) || optype.equals(Opcodes.MINUS.toString()) //sparse safe
						&& ( leftSparse || rightSparse ) )
						return d1m*d1n*d1s + d2m*d2n*d2s;
					else if( optype.equals(Opcodes.SOLVE.toString()) ) //see also MultiReturnBuiltin
						return d1m * d1n * d1n; //for 1kx1k ~ 1GFLOP -> 0.5s
					else
						return d3m*d3n;
				
				case Ternary: //opcodes: +*, -*, ifelse
					return 2 * d1m * d1n;
					
				case Ctable: //opcodes: ctable
					if( optype.equals(Opcodes.CTABLE.toString()) ){
						if( leftSparse )
							return d1m * d1n * d1s; //add
						else 
							return d1m * d1n;
					}
					return 0;
				
				case Builtin: //opcodes: log 
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					//note: can be unary or binary
					if( allExists ) //binary
						return 3 * d3m * d3n;
					else //unary
						return d3m * d3n;
					
				case Unary: //opcodes: exp, abs, sin, cos, tan, sign, sqrt, plogp, print, round, sprop, sigmoid
					//TODO add cost functions for commons math builtins: inverse, cholesky
					if( optype.equals(Opcodes.PRINT.toString()) ) //scalar only
						return 1;
					else
					{
						double xbu = 1; //default for all ops
						if( optype.equals(Opcodes.PLOGP.toString()) ) xbu = 2;
						else if( optype.equals(Opcodes.ROUND.toString()) ) xbu = 4;
						
						if( optype.equals(Opcodes.SIN.toString()) || optype.equals(Opcodes.TAN.toString()) || optype.equals(Opcodes.ROUND.toString())
							|| optype.equals(Opcodes.ABS.toString()) || optype.equals(Opcodes.SQRT.toString()) || optype.equals(Opcodes.SPROP.toString())
							|| optype.equals(Opcodes.SIGMOID.toString()) || optype.equals(Opcodes.SIGN.toString()) ) //sparse-safe
						{
							if( leftSparse ) //sparse
								return xbu * d1m * d1n * d1s;	
							else //dense
								return xbu * d1m * d1n;
						}
						else
							return xbu * d1m * d1n;
					}

				case Reorg: //opcodes: r', rdiag
				case Reshape: //opcodes: rshape
					if( leftSparse )
						return d1m * d1n * d1s;
					else
						return d1m * d1n;
					
				case Append: //opcodes: append
					return DEFAULT_NFLOP_CP * 
					       (((leftSparse) ? d1m * d1n * d1s : d1m * d1n ) +
					        ((rightSparse) ? d2m * d2n * d2s : d2m * d2n ));
				
				case Variable: //opcodes: assignvar, cpvar, rmvar, rmfilevar, assignvarwithfile, attachfiletovar, valuepick, iqsize, read, write, createvar, setfilename, castAsMatrix
					if( optype.equals(Opcodes.WRITE.toString()) ){
						FileFormat fmt = FileFormat.safeValueOf(args[0]);
						boolean text = fmt.isTextFormat();
						double xwrite =  text ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
						
						if( !leftSparse )
							return d1m * d1n * xwrite; 
						else
							return d1m * d1n * d1s * xwrite;
					}
					else if ( optype.equals("inmem-iqm") )
						//note: assumes uniform distribution
						return 2 * d1m + //sum of weights
						       5 + 0.25d * d1m + //scan to lower quantile
						       8 * 0.5 * d1m; //scan from lower to upper quantile
					else
						return DEFAULT_NFLOP_NOOP;
			
				case Rand: //opcodes: rand, seq
					if( optype.equals(Opcodes.RANDOM.toString()) ){
						int nflopRand = 32; //per random number
						switch(Integer.parseInt(args[0])) {
							case 0: return DEFAULT_NFLOP_NOOP; //empty matrix
							case 1: return d3m * d3n * 8; //allocate, arrayfill
							case 2: //full rand
							{
								if( d3s==1.0 )
									return d3m * d3n * nflopRand + d3m * d3n * 8; //DENSE gen (incl allocate)    
								else 
									return (d3s>=MatrixBlock.SPARSITY_TURN_POINT)? 
										    2 * d3m * d3n * nflopRand + d3m * d3n * 8: //DENSE gen (incl allocate)    
									        3 * d3m * d3n * d3s * nflopRand + d3m * d3n * d3s * 24; //SPARSE gen (incl allocate)
							}
						}
					}
					else //seq
						return d3m * d3n * DEFAULT_NFLOP_CP;
				
				case StringInit: //sinit
					return d3m * d3n * DEFAULT_NFLOP_CP;
					
				case FCall: //opcodes: fcall
					//note: should be invoked independently for multiple outputs
					return d1m * d1n * d1s * DEFAULT_NFLOP_UNKNOWN;
				
				case MultiReturnBuiltin: //opcodes: qr, lu, eigen, svd
					//note: they all have cubic complexity, the scaling factor refers to commons.math
					double xf = 2; //default e.g, qr
					if( optype.equals(Opcodes.EIGEN.toString()) )
						xf = 32;
					else if ( optype.equals(Opcodes.LU.toString()) )
						xf = 16;
					else if ( optype.equals(Opcodes.SVD.toString()))
						xf = 32;	// TODO - assuming worst case for now
					return xf * d1m * d1n * d1n; //for 1kx1k ~ 2GFLOP -> 1s
					
				case ParameterizedBuiltin: //opcodes: cdf, invcdf, groupedagg, rmempty
					if( optype.equals(Opcodes.CDF.toString()) || optype.equals(Opcodes.INVCDF.toString()))
						return DEFAULT_NFLOP_UNKNOWN; //scalar call to commons.math
					else if( optype.equals(Opcodes.GROUPEDAGG.toString()) ){
						double xga = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xga=4; break; //sum, see uk+
							case 1: xga=1; break; //count, see cm
							case 2: xga=8; break; //mean
							case 3: xga=16; break; //cm2
							case 4: xga=31; break; //cm3
							case 5: xga=51; break; //cm4
							case 6: xga=16; break; //variance
						}						
						return 2 * d1m + xga * d1m; //scan for min/max, groupedagg
					}	
					else if( optype.equals(Opcodes.RMEMPTY.toString()) ){
						switch(Integer.parseInt(args[0])){
							case 0: //remove rows
								return ((leftSparse) ? d1m : d1m * Math.ceil(1.0d/d1s)/2) +
									   DEFAULT_NFLOP_CP * d3m * d2m;
							case 1: //remove cols
								return d1n * Math.ceil(1.0d/d1s)/2 + 
								       DEFAULT_NFLOP_CP * d3m * d2m;
						}
						
					}	
					return 0;
					
				case QSort: //opcodes: sort
					if( optype.equals("sort") ){
						//note: mergesort since comparator used
						double sortCosts = 0;
						if( onlyLeft )
							sortCosts = DEFAULT_NFLOP_CP * d1m + d1m;
						else //w/ weights
							sortCosts = DEFAULT_NFLOP_CP * ((leftSparse)?d1m*d1s:d1m); 
						return sortCosts + d1m*(int)(Math.log(d1m)/Math.log(2)) + //mergesort
										   DEFAULT_NFLOP_CP * d1m;
					}
					return 0;
					
				case MatrixIndexing: //opcodes: rightIndex, leftIndex
					if( optype.equals(Opcodes.LEFT_INDEX.toString()) ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n)
						       + 2 * DEFAULT_NFLOP_CP * ((rightSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					else if( optype.equals(Opcodes.RIGHT_INDEX.toString()) ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					return 0;
					
				case MMTSJ: //opcodes: tsmm
					//diff to ba+* only upper triangular matrix
					//reduction by factor 2 because matrix mult better than
					//average flop count
					if( MMTSJType.valueOf(args[0]).isLeft() ) { //lefttranspose
						if( !rightSparse ) //dense
							return d1m * d1n * d1s * d1n /2;
						else //sparse
							return d1m * d1n * d1s * d1n * d1s /2; 
					}
					else if(onlyLeft) { //righttranspose
						if( !leftSparse ) //dense
							return (double)d1m * d1n * d1m /2;
						else //sparse
							return   d1m * d1n * d1s //reorg sparse
							       + d1m * d1n * d1s * d1n * d1s /2; //core tsmm
					}
					return 0;
				
				case Partition:
					return d1m * d1n * d1s + //partitioning costs
						   (inMR ? 0 : //include write cost if in CP
							getHDFSWriteTime(d1m, d1n, d1s)* DEFAULT_FLOPS);
				
				default: 
					throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
			}
		}
		else
		{
			throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
		}
	}
}
