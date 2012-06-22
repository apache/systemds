package com.ibm.bi.dml.runtime.instructions.CPInstructions;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public class CM_COV_Object extends Data {

	//for central moment
	public double w;
	public KahanObject mean;
	public KahanObject m2;
	public KahanObject m3;
	public KahanObject m4;
	
	public KahanObject mean_v;
	public KahanObject c2;
	
/*	public double mean;
	public double mean_correction;
	public double m2;
	public double m3;
	public double m4;
	
	//for cov
	public double mean_v;
	public double mean_v_correction;
	public double c2;*/
	
	public String toString()
	{
		return "weight: "+w+", mean: "+mean+", m2: "+m2+", m3: "+m3+", m4: "+m4+", mean2: "+mean_v+", c2: "+c2;
	}
	
	public CM_COV_Object()
	{
		super(DataType.OBJECT, ValueType.UNKNOWN);
		w=0;
		mean=new KahanObject(0,0);
		m2=new KahanObject(0,0);
		m3=new KahanObject(0,0);
		m4=new KahanObject(0,0);
		mean_v=new KahanObject(0,0);
		c2=new KahanObject(0,0);
		/*
		w=mean=m2=m3=m4=mean_v=c2=0;
		mean_correction=0;
		mean_v_correction=0;*/
	}
	
	public void reset()
	{
		w=0;
		mean=new KahanObject(0,0);
		m2=new KahanObject(0,0);
		m3=new KahanObject(0,0);
		m4=new KahanObject(0,0);
		mean_v=new KahanObject(0,0);
		c2=new KahanObject(0,0);
	}
	
	public int compareTo(CM_COV_Object that)
	{
		if(w!=that.w)
			return Double.compare(w, that.w);
		else if(mean!=that.mean)
			return KahanObject.compare(mean, that.mean);
		else if(m2!=that.m2)
			return KahanObject.compare(m2, that.m2);
		else if(m3!=that.m3)
			return KahanObject.compare(m3, that.m3);
		else if(m4!=that.m4)
			return KahanObject.compare(m4, that.m4);
		else if(mean_v!=that.mean_v)
			return KahanObject.compare(mean_v, that.mean_v);
		else
			return KahanObject.compare(c2, that.c2);
	}
	
	public void set(CM_COV_Object that)
	{
		this.w=that.w;
		this.mean.set(that.mean);
		this.m2.set(that.m2);
		this.m3.set(that.m3);
		this.m4.set(that.m4);
		this.mean_v.set(that.mean_v);
		this.c2.set(that.c2);
		//this.mean_correction=that.mean_correction;
		//this.mean_v_correction=that.mean_v_correction;
	}
	
	public boolean isCMAllZeros()
	{
		return w==0 && mean.isAllZero() && m2.isAllZero()  && m3.isAllZero()  && m4.isAllZero() ;
	}
	
	public boolean isCOVAllZeros()
	{
		return w==0 && mean.isAllZero()  && mean_v.isAllZero() && c2.isAllZero() ;
	}
	
	public double getRequiredResult(Operator op) throws DMLRuntimeException
	{
		if(op instanceof CMOperator)
		{
			AggregateOperationTypes agg=((CMOperator)op).aggOpType;
			switch(agg)
			{
			case COUNT:
				return w;
			case MEAN:
				return mean._sum;
			case CM2:
				return m2._sum/w;
			case CM3:
				return m3._sum/w;
			case CM4:
				return m4._sum/w;
			case VARIANCE:
				return w==1.0? 0:m2._sum/(w-1);
			default:
				throw new DMLRuntimeException("Invalid aggreagte in CM_CV_Object: " + agg);
			}
		}else
		{
			//avoid division by 0
			if(w==1.0)
				return 0;
			else
				return c2._sum/(w-1.0);
		}
	}

	@Override
	public String getDebugName() {
		// TODO Auto-generated method stub
		return null;
	}
}
