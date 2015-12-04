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

package org.apache.sysml.lops;

import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform a partial aggregation. It was introduced to do some initial
 * aggregation operations on blocks in the mapper/reducer.
 */

public class PartialAggregate extends Lop 
{

	
	public enum DirectionTypes {
		RowCol, 
		Row, 
		Col
	};

	public enum CorrectionLocationType { 
		NONE, 
		LASTROW, 
		LASTCOLUMN, 
		LASTTWOROWS, 
		LASTTWOCOLUMNS, 
		INVALID 
	};
	
	private Aggregate.OperationTypes operation;
	private DirectionTypes direction;
	private boolean _dropCorr = false;
	
	//optional attribute for CP num threads
	private int _numThreads = -1;
	
	//optional attribute for spark exec type
	private SparkAggType _aggtype = SparkAggType.MULTI_BLOCK;
	
	public PartialAggregate( Lop input, Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt)
		throws LopsException 
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, ExecType.MR);
	}

	public PartialAggregate( Lop input, Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, et);
	}
	
	public PartialAggregate( Lop input, Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt, ExecType et, int k)
		throws LopsException 
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, et);
		_numThreads = k;
	}
	
	public PartialAggregate( Lop input, Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt, SparkAggType aggtype, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.PartialAggregate, dt, vt);
		init(input, op, direct, dt, vt, et);
		_aggtype = aggtype;
	}
	
	/**
	 * Constructor to setup a partial aggregate operation.
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */
	private void init(Lop input,
			Aggregate.OperationTypes op,
			PartialAggregate.DirectionTypes direct, DataType dt, ValueType vt, ExecType et) {
		operation = op;
		direction = direct;
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) 
		{
			/*
			 * This lop CAN NOT be executed in PARTITION, SORT, STANDALONE MMCJ:
			 * only in mapper.
			 */
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			this.lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		} 
		else //CP | SPARK
		{
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}

	public void setDropCorrection()
	{
		_dropCorr = true;
	}
	
	public static CorrectionLocationType decodeCorrectionLocation(String loc) 
	{
		return CorrectionLocationType.valueOf(loc);
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
	 */
	public CorrectionLocationType getCorrectionLocation() 
		throws LopsException 
	{
		return getCorrectionLocation(operation, direction);
	}
	
	public static CorrectionLocationType getCorrectionLocation(Aggregate.OperationTypes operation, DirectionTypes direction) 
		throws LopsException 
	{
		CorrectionLocationType loc;

		switch (operation) {
		case KahanSum:
		case KahanSumSq:
		case KahanTrace:
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

		case Mean:
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
			
		case MaxIndex:
		case MinIndex:
			loc = CorrectionLocationType.LASTCOLUMN;
			break;
			
		default:
			loc = CorrectionLocationType.NONE;
		}
		return loc;
	}

	public void setDimensionsBasedOnDirection(long dim1, long dim2,  
			long rowsPerBlock, long colsPerBlock) throws LopsException 
	{
		setDimensionsBasedOnDirection(this, dim1, dim2, rowsPerBlock, colsPerBlock, direction);
	}

	public static void setDimensionsBasedOnDirection(Lop lop, long dim1, long dim2,  
			long rowsPerBlock, long colsPerBlock, DirectionTypes dir) 
		throws LopsException 
	{
		try {
			if (dir == DirectionTypes.Row)
				lop.outParams.setDimensions(dim1, 1, rowsPerBlock, colsPerBlock, -1);
			else if (dir == DirectionTypes.Col)
				lop.outParams.setDimensions(1, dim2, rowsPerBlock, colsPerBlock, -1);
			else if (dir == DirectionTypes.RowCol)
				lop.outParams.setDimensions(1, 1, rowsPerBlock, colsPerBlock, -1);
			else
				throw new LopsException("In PartialAggregate Lop, Unknown aggregate direction " + dir);
		} catch (HopsException e) {
			throw new LopsException("In PartialAggregate Lop, error setting dimensions based on direction", e);
		}
	}
	
	public String toString() {
		return "Partial Aggregate " + operation;
	}
	
	private String getOpcode() {
		return getOpcode(operation, direction);
	}

	/**
	 * Instruction generation for for CP and Spark
	 */
	@Override
	public String getInstructions(String input1, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output) );
		
		//in case of spark, we also compile the optional aggregate flag into the instruction.
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _aggtype );	
		}
		
		//in case of cp, we also compile the number of threads into the instruction
		if( getExecType() == ExecType.CP ){
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );	
		}
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index)
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _dropCorr );

		return sb.toString();
	}

	/**
	 * 
	 * @param op
	 * @param dir
	 * @return
	 */
	public static String getOpcode(Aggregate.OperationTypes op, DirectionTypes dir) 
	{
		switch( op )
		{
			case Sum: {
				if( dir == DirectionTypes.RowCol ) 
					return "ua+";
				else if( dir == DirectionTypes.Row ) 
					return "uar+";
				else if( dir == DirectionTypes.Col ) 
					return "uac+";
				break;
			}

			case Mean: {
				if( dir == DirectionTypes.RowCol ) 
					return "uamean";
				else if( dir == DirectionTypes.Row ) 
					return "uarmean";
				else if( dir == DirectionTypes.Col ) 
					return "uacmean";
				break;
			}			
			
			case KahanSum: {
				// instructions that use kahanSum are similar to ua+,uar+,uac+
				// except that they also produce correction values along with partial
				// sums.
				if( dir == DirectionTypes.RowCol )
					return "uak+";
				else if( dir == DirectionTypes.Row )
					return "uark+";
				else if( dir == DirectionTypes.Col ) 
					return "uack+";
				break;
			}

			case KahanSumSq: {
				if( dir == DirectionTypes.RowCol )
					return "uasqk+";
				else if( dir == DirectionTypes.Row )
					return "uarsqk+";
				else if( dir == DirectionTypes.Col )
					return "uacsqk+";
				break;
			}

			case Product: {
				if( dir == DirectionTypes.RowCol )
					return "ua*";
				break;
			}
			
			case Max: {
				if( dir == DirectionTypes.RowCol ) 
					return "uamax";
				else if( dir == DirectionTypes.Row ) 
					return "uarmax";
				else if( dir == DirectionTypes.Col )
					return "uacmax";
				break;
			}
			
			case Min: {
				if( dir == DirectionTypes.RowCol ) 
					return "uamin";
				else if( dir == DirectionTypes.Row ) 
					return "uarmin";
				else if( dir == DirectionTypes.Col ) 
					return "uacmin";
				break;
			}
			
			case MaxIndex:{
				if( dir == DirectionTypes.Row )
					return "uarimax";
				break;
			}
			
			case MinIndex: {
				if( dir == DirectionTypes.Row )
					return "uarimin";
				break;
			}
		
			case Trace: {
				if( dir == DirectionTypes.RowCol)
					return "uatrace";
				break;	
			}
			
			case KahanTrace: {
				if( dir == DirectionTypes.RowCol ) 
					return "uaktrace";
				break;
			}
		}
		
		//should never come here for normal compilation
		throw new UnsupportedOperationException("Instruction is not defined for PartialAggregate operation " + op);
	}

}
