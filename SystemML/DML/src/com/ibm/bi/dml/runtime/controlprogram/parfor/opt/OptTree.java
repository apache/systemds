package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

/**
 * Represents a complete plan of a top-level parfor. This includes the internal
 * representation of the actual current plan as well as additional meta information 
 * that are only kept once per program instead of for each and every plan alternative.
 * 
 */
public class OptTree 
{
	//global contraints 
	private int     _ck;  //max constraint degree of parallelism
	private double  _cm;  //max constraint memory consumption
	
	//actual tree
	private OptNode _root;
	
	//internal node metadata repository
	//private HashMap<Integer, Integer> _kcp;
	//private HashMap<Integer, Integer> _kmr;
	
	public OptTree( int ck, double cm, OptNode node )
	{
		_ck = ck;
		_cm = cm;
		
		_root = node;
	}
	
	///////
	// getter and setter
	
	public int getCK()
	{
		return _ck;
	}
	
	public double getCM()
	{
		return _cm;
	}
	
	public OptNode getRoot()
	{
		return _root;
	}
	
	public void setRoot( OptNode n )
	{
		_root = n;
	}
	
	/**
	 * Explain tool: prints the hierarchical plan (including all available 
	 * detail information, if necessary) to <code>stdout</code>.
	 * 
	 * @param withDetails
	 * @return
	 */
	public String explain( boolean withDetails )
	{
		StringBuilder sb = new StringBuilder();
		
		sb.append(" OPT TREE (size=");
		sb.append(_root.size());
		sb.append(")\n");
		sb.append("---------------------\n");
		sb.append(_root.explain(1, withDetails));
		sb.append("---------------------\n");
		
		return sb.toString();
	}
}
