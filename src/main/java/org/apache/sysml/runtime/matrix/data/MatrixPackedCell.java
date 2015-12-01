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


package org.apache.sysml.runtime.matrix.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;


public class MatrixPackedCell extends MatrixCell
{

	private static final long serialVersionUID = -3633665169444817750L;

	private double[] extras=null;
	private int extra_size=0;
	
	public MatrixPackedCell(double v, int n)
	{
		value=v;
		checkAndAllocateSpace(n);
	}
	
	public MatrixPackedCell(MatrixPackedCell that)
	{
		this.value=that.value;
		checkAndAllocateSpace(that.extra_size);
		for(int i=0; i<extra_size; i++)
			extras[i]=that.extras[i];
	}
	
	public MatrixPackedCell() {
		super();
	}

	private void checkAndAllocateSpace(int size)
	{
		if(extras==null || extras.length<size)
			extras=new double[size];
		extra_size=size;
	}
	
	public static MatrixPackedCell checkType(MatrixValue cell) throws DMLUnsupportedOperationException
	{
		if( cell!=null && !(cell instanceof MatrixPackedCell))
			throw new DMLUnsupportedOperationException("the Matrix Value is not MatrixPackedCell!");
		return (MatrixPackedCell) cell;
	}
	
	public double getExtraByPostition(int i)
	{
		if(extras==null || i>=extra_size)
			return 0;
		else
			return extras[i];
	}

	//with corrections
	@Override
	public void incrementalAggregate(AggregateOperator aggOp, MatrixValue correction, 
			MatrixValue newWithCorrection)throws DMLUnsupportedOperationException, DMLRuntimeException {
		incrementalAggregate(aggOp, newWithCorrection);
	}
	
	//with corrections
	@Override
	public void incrementalAggregate(AggregateOperator aggOp,
			MatrixValue newWithCorrection)throws DMLUnsupportedOperationException, DMLRuntimeException {

		
		MatrixPackedCell newWithCor=checkType(newWithCorrection);
		if(aggOp.correctionLocation==CorrectionLocationType.NONE || aggOp.correctionLocation==CorrectionLocationType.LASTROW || aggOp.correctionLocation==CorrectionLocationType.LASTCOLUMN)
		{
			checkAndAllocateSpace(1);
			KahanObject buffer=new KahanObject(value, extras[0]);
			buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, newWithCor.value, newWithCor.getExtraByPostition(0));
			value=buffer._sum;
			extras[0]=buffer._correction;
		//	System.out.println("--- "+buffer);
		}else if(aggOp.correctionLocation==CorrectionLocationType.LASTROW || aggOp.correctionLocation==CorrectionLocationType.LASTTWOCOLUMNS)
		{
			checkAndAllocateSpace(2);
			KahanObject buffer=new KahanObject(value, extras[0]);
			buffer._sum=value;
			double n=extras[0];
			buffer._correction=extras[1];
			double mu2=newWithCor.value;
			double n2=newWithCor.getExtraByPostition(0);
			n=n+n2;
			double toadd=(mu2-buffer._sum)*n2/n;
			buffer=(KahanObject) aggOp.increOp.fn.execute(buffer, toadd);
			value=buffer._sum;
			extras[0]=n;
			extras[1]=buffer._correction;
		}
		else
			throw new DMLRuntimeException("unrecognized correctionLocation: "+aggOp.correctionLocation);
		
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		value=in.readDouble();
		int n=in.readInt();
		if(extras==null || extras.length<n)
			extras=new double[n];
		for(int i=0; i<n; i++)
			extras[i]=in.readDouble();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(value);
		out.writeInt(extra_size);
		for(int i=0; i<extra_size; i++)
			out.writeDouble(extras[i]);
	}

	@Override
	public int compareTo(Object other) {
		
		if(!(other instanceof MatrixPackedCell))
			throw new RuntimeException("cannot compare MatrixPackedCell with "+other.getClass());
		MatrixPackedCell that=(MatrixPackedCell) other;
		if(this.value!=that.value)
			return Double.compare(this.value, that.value);
		else if(this.extra_size!=that.extra_size)
			return this.extra_size-that.extra_size;
		else
		{
			for(int i=0; i<extra_size; i++)
			{
				if(this.extras[i]!=that.extras[i])
					return Double.compare(this.extras[i], that.extras[i]);
			}
			return 0;
		}
	}
	
	@Override
	public boolean equals(Object other) {
		
		if(!(other instanceof MatrixPackedCell))
			throw new RuntimeException("cannot compare MatrixPackedCell with "+other.getClass());
		
		MatrixPackedCell that=(MatrixPackedCell) other;
		boolean ret = (value==that.value && extra_size==that.extra_size);
		if( ret ) {
			for(int i=0; i<extra_size; i++)
				if(extras[i]!=that.extras[i])
					return false;
		}
		
		return ret;
	}
	
	@Override
	public int hashCode() {
		throw new RuntimeException("hashCode() should never be called on instances of this class.");
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("\nextras: ");
		for(int i=0; i<extra_size; i++){
			sb.append(extras[i]);
			sb.append(", ");
		}
		
		return sb.toString();
	}
}
