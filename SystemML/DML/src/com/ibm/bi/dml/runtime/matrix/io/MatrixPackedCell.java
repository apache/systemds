package com.ibm.bi.dml.runtime.matrix.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.KahanObject;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;



public class MatrixPackedCell extends MatrixCell{

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
	
	public String toString()
	{
		String str= super.toString()+"\nextras: ";
		for(int i=0; i<extra_size; i++)
			str+=(extras[i]+", ");
		return str;
	}
	
}
