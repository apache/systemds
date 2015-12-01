/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.*;


/**
 * Lop to perform transpose-identity operation (t(X)%*%X or X%*%t(X)),
 * used to represent CP and MR instruction but in case of MR there is
 * an additional Aggregate at the reducers.
 */
public class MMTSJ extends Lop 
{

	
	public enum MMTSJType {
		NONE,
		LEFT,
		RIGHT;
		
		public boolean isLeft(){
			return (this == LEFT);
		}
	}
	
	private MMTSJType _type = null;
	private int _numThreads = 1;

	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type) 
	{
		this(input1, dt, vt, et, type, -1);
	}
	
	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type, int k) 
	{
		super(Lop.Type.MMTSJ, dt, vt);		
		addInput(input1);
		input1.addOutput(this);
		_type = type;
		_numThreads = k;
		 
		boolean breaksAlignment = true; //if result keys (matrix indexes) different 
		boolean aligner = false; //if groups multiple inputs by key (e.g., group)
		boolean definesMRJob = (et == ExecType.MR); //if requires its own MR job 
		ExecLocation el = (et == ExecType.MR) ? ExecLocation.Map : ExecLocation.ControlProgram;
		
		lps.addCompatibility(JobType.GMR);
		lps.setProperties( inputs, et, el, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
		return "Operation = MMTSJ";
	}

	/**
	 * MR instruction generation.
	 */
	@Override
	public String getInstructions(int input_index1, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "tsmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index));
		sb.append( OPERAND_DELIMITOR );
		sb.append( _type );
		
		return sb.toString();
	}

	/**
	 * CP and Spark instruction generation.
	 */
	@Override
	public String getInstructions(String input_index1, String output_index) throws LopsException
	{	
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "tsmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index));
		sb.append( OPERAND_DELIMITOR );
		sb.append( _type );
		
		//append degree of parallelism for matrix multiplications
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		
		return sb.toString();
	}
 
 
}