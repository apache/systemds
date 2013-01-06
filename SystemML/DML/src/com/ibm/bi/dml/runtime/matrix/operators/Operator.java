package com.ibm.bi.dml.runtime.matrix.operators;

public class Operator 
{
	public boolean sparseSafe=false;

	public Operator()
	{
		
	}
	
	public Operator(boolean sparseSafeFlag)
	{
		sparseSafe = sparseSafeFlag;
	}
}
