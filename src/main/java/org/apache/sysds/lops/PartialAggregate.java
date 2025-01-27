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

package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

/**
 * Lop to perform a partial aggregation. It was introduced to do some initial
 * aggregation operations on blocks in the mapper/reducer.
 */

public class PartialAggregate extends Lop 
{
	private AggOp operation;
	private Direction direction;
	
	//optional attribute for CP num threads
	private int _numThreads = -1;
	
	//optional attribute for spark exec type
	private SparkAggType _aggtype = SparkAggType.MULTI_BLOCK;

	public PartialAggregate( Lop input, AggOp op, Direction direct, DataType dt, ValueType vt, ExecType et, int k)
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, et);
		_numThreads = k;
	}
	
	public PartialAggregate( Lop input, AggOp op, Direction direct, DataType dt, ValueType vt, SparkAggType aggtype, ExecType et)
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, et);
		_aggtype = aggtype;
	}
	
	/**
	 * Constructor to setup a partial aggregate operation.
	 * 
	 * @param input low-level operator
	 * @param op aggregate operation type
	 * @param direct partial aggregate directon type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	private void init(Lop input, AggOp op, Direction direct, DataType dt, ValueType vt, ExecType et) {
		operation = op;
		direction = direct;
		addInput(input);
		input.addOutput(this);
		lps.setProperties(inputs, et);
	}

	/**
	 * This method computes the location of "correction" terms in the output
	 * produced by PartialAgg instruction.
	 * 
	 * When computing the stable sum, "correction" refers to the compensation as
	 * defined by the original Kahan algorithm. When computing the stable mean,
	 * "correction" refers to two extra values (the running mean, count)
	 * produced by each Mapper i.e., by each PartialAgg instruction.
	 * 
	 * This method is invoked during hop-to-lop translation, while creating the
	 * corresponding Aggregate lop
	 * 
	 * Computed information is encoded in the PartialAgg instruction so that the
	 * appropriate aggregate operator is used at runtime (see:
	 * dml.runtime.matrix.operator.AggregateOperator.java and dml.runtime.matrix)
	 * 
	 * @return correct location
	 */
	public CorrectionLocationType getCorrectionLocation() {
		return getCorrectionLocation(operation, direction);
	}
	
	public static CorrectionLocationType getCorrectionLocation(AggOp operation, Direction direction) {
		CorrectionLocationType loc;

		switch (operation) {
		case SUM:
		case SUM_SQ:
		case TRACE:
			switch (direction) {
				case Col:
					// colSums: corrections will be present as a last row in the
					// result
					loc = CorrectionLocationType.LASTROW;
					break;
				case Row:
				case RowCol:
					// rowSums, sum: corrections will be present as a last column in
					// the result
					loc = CorrectionLocationType.LASTCOLUMN;
					break;
				default:
					throw new LopsException("PartialAggregate.getCorrectionLocation() - "
										+ "Unknown aggregate direction: " + direction);
			}
			break;

		case MEAN:
			// Computation of stable mean requires each mapper to output both
			// the running mean as well as the count
			switch (direction) {
				case Col:
					// colMeans: last row is correction 2nd last is count
					loc = CorrectionLocationType.LASTTWOROWS;
					break;
				case Row:
				case RowCol:
					// rowMeans, mean: last column is correction 2nd last is count
					loc = CorrectionLocationType.LASTTWOCOLUMNS;
					break;
				default:
					throw new LopsException("PartialAggregate.getCorrectionLocation() - "
							+ "Unknown aggregate direction: " + direction);
			}
			break;

		case VAR:
			// Computation of stable variance requires each mapper to
			// output the running variance, the running mean, the
			// count, a correction term for the squared deviations
			// from the sample mean (m2), and a correction term for
			// the mean.  These values collectively allow all other
			// necessary intermediates to be reconstructed, and the
			// variance will output by our unary aggregate framework.
			// Thus, our outputs will be:
			// { var | mean, count, m2 correction, mean correction }
			switch (direction) {
				case Col:
					// colVars: { var | mean, count, m2 correction, mean correction },
					// where each element is a column.
					loc = CorrectionLocationType.LASTFOURROWS;
					break;
				case Row:
				case RowCol:
					// var, rowVars: { var | mean, count, m2 correction, mean correction },
					// where each element is a row.
					loc = CorrectionLocationType.LASTFOURCOLUMNS;
					break;
				default:
					throw new LopsException("PartialAggregate.getCorrectionLocation() - "
							+ "Unknown aggregate direction: " + direction);
			}
			break;

		case MAXINDEX:
		case MININDEX:
			loc = CorrectionLocationType.LASTCOLUMN;
			break;
			
		default:
			loc = CorrectionLocationType.NONE;
		}
		return loc;
	}
	
	@Override
	public SparkAggType getAggType() {
		return _aggtype;
	}

	public void setDimensionsBasedOnDirection(long dim1, long dim2, long blen) {
		setDimensionsBasedOnDirection(this, dim1, dim2, blen, direction);
	}

	public static void setDimensionsBasedOnDirection(Lop lop, long dim1, long dim2,  long blen, Direction dir)
	{
		try {
			if (dir == Direction.Row)
				lop.outParams.setDimensions(dim1, 1, blen, -1);
			else if (dir == Direction.Col)
				lop.outParams.setDimensions(1, dim2, blen, -1);
			else if (dir == Direction.RowCol)
				lop.outParams.setDimensions(1, 1, blen, -1);
			else
				throw new LopsException("In PartialAggregate Lop, Unknown aggregate direction " + dir);
		} catch (HopsException e) {
			throw new LopsException("In PartialAggregate Lop, error setting dimensions based on direction", e);
		}
	}
	
	@Override
	public String toString() {
		return "Partial Aggregate " + operation;
	}
	
	private String getOpcode() {
		return getOpcode(operation, direction);
	}

	/**
	 * Instruction generation for CP and Spark
	 */
	@Override
	public String getInstructions(String input1, String output) 
	{
		String ret = InstructionUtils.concatOperands(
			getExecType().name(), getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			prepOutputOperand(output));

		if ( getExecType() == ExecType.SPARK )
			ret = InstructionUtils.concatOperands(ret, _aggtype.name());
		else if ( getExecType() == ExecType.CP || getExecType() == ExecType.FED ){
			ret = InstructionUtils.concatOperands(ret, Integer.toString(_numThreads));
			if ( getOpcode().equalsIgnoreCase(Opcodes.UARIMIN.toString()) || getOpcode().equalsIgnoreCase(Opcodes.UARIMAX.toString()) )
				ret = InstructionUtils.concatOperands(ret, "1");
			if ( getExecType() == ExecType.FED )
				ret = InstructionUtils.concatOperands(ret, _fedOutput.name());
		}
		
		return ret;
	}

	public static String getOpcode(AggOp op, Direction dir)
	{
		switch( op )
		{
			case SUM: {
				// instructions that use kahanSum are similar to ua+,uar+,uac+
				// except that they also produce correction values along with partial
				// sums.
				if( dir == Direction.RowCol )
					return Opcodes.UAKP.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARKP.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACKP.toString();
				break;
			}

			case SUM_SQ: {
				if( dir == Direction.RowCol )
					return Opcodes.UASQKP.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARSQKP.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACSQKP.toString();
				break;
			}

			case MEAN: {
				if( dir == Direction.RowCol )
					return Opcodes.UAMEAN.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARMEAN.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACMEAN.toString();
				break;
			}

			case VAR: {
				if( dir == Direction.RowCol )
					return Opcodes.UAVAR.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARVAR.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACVAR.toString();
				break;
			}

			case PROD: {
				switch( dir ) {
					case RowCol: return Opcodes.UAM.toString();
					case Row:    return Opcodes.UARM.toString();
					case Col:    return Opcodes.UACM.toString();
				}
			}
			
			case SUM_PROD: {
				switch( dir ) {
					case RowCol: return "ua+*";
					case Row:    return "uar+*";
					case Col:    return "uac+*";
				}
			}
			
			case MAX: {
				if( dir == Direction.RowCol )
					return Opcodes.UAMAX.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARMAX.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACMAX.toString();
				break;
			}
			
			case MIN: {
				if( dir == Direction.RowCol )
					return Opcodes.UAMIN.toString();
				else if( dir == Direction.Row )
					return Opcodes.UARMIN.toString();
				else if( dir == Direction.Col )
					return Opcodes.UACMIN.toString();
				break;
			}
			
			case MAXINDEX:{
				if( dir == Direction.Row )
					return Opcodes.UARIMAX.toString();
				break;
			}
			
			case MININDEX: {
				if( dir == Direction.Row )
					return Opcodes.UARIMIN.toString();
				break;
			}
			
			case TRACE: {
				if( dir == Direction.RowCol )
					return Opcodes.UAKTRACE.toString();
				break;
			}

			case COUNT_DISTINCT: {
				switch (dir) {
					case RowCol: return Opcodes.UACD.toString();
					case Row: return Opcodes.UACDR.toString();
					case Col: return Opcodes.UACDAPC.toString();
					default:
						throw new LopsException("PartialAggregate.getOpcode() - "
								+ "Unknown aggregate direction: " + dir);
				}
			}

			case COUNT_DISTINCT_APPROX: {
				switch (dir) {
					case RowCol: return Opcodes.UACDAP.toString();
					case Row: return Opcodes.UACDAPR.toString();
					case Col: return Opcodes.UACDAPC.toString();
					default:
						throw new LopsException("PartialAggregate.getOpcode() - "
								+ "Unknown aggregate direction: " + dir);
				}
			}

			case UNIQUE: {
				switch (dir) {
					case RowCol: return Opcodes.UNIQUE.toString();
					case Row: return Opcodes.UNIQUER.toString();
					case Col: return Opcodes.UNIQUEC.toString();
					default:
						throw new LopsException("PartialAggregate.getOpcode() - "
								+ "Unknown aggregate direction: " + dir);
				}
			}
		}
		
		//should never come here for normal compilation
		throw new UnsupportedOperationException("Instruction is not defined for PartialAggregate operation " + op);
	}

}
