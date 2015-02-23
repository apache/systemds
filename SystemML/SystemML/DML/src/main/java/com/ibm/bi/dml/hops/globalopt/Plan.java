package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;
import com.ibm.bi.dml.lops.LopProperties.ExecType;


public class Plan 
{
	private InterestingProperties _ips = null;
	private RewriteConfig _conf        = null;
	private ArrayList<Plan> _childs    = null;
	private double _costs = -1;
	
	public Plan(InterestingProperties ips, RewriteConfig rc, ArrayList<Plan> childs)
	{
		_ips = ips;
		_conf = rc;
		if( childs != null && !childs.isEmpty() )
			_childs = childs;
		else
			_childs = new ArrayList<Plan>();
	}
	
	public Plan( Plan p )
	{	
		_ips = new InterestingProperties(p._ips);
		_conf = new RewriteConfig(p._conf);
		_costs = p._costs;
		
		if( p._childs != null && !p._childs.isEmpty() )
			_childs = (ArrayList<Plan>) p._childs.clone();
		else
			_childs = new ArrayList<Plan>();
			
	}
	
	public void addChild( Plan c )
	{
		_childs.add(c);
	}
	
	public InterestingProperties getInterestingProperties()
	{
		return _ips;
	}
	
	public void setCosts( double costs )
	{
		_costs = costs;
	}
	
	public double getCosts()
	{
		return _costs;
	}
	
	/**
	 * If operation is executed in MR, all input blocksizes need to match.
	 * Note that the output blocksize can be different since we would add
	 * additional reblocks after that operation.
	 * 
	 * @return
	 */
	public boolean checkValidBlocksizesInMR()
	{
		boolean ret = true;
		
		if(    _conf.getExecType()==ExecType.MR 
			&& _childs != null && _childs.size() > 1 ) 
		{
			int size0 = _childs.get(0)._conf.getBlockSize();
			for( Plan c : _childs )
				ret &= (c._conf.getBlockSize() == size0);
		}
		
		return ret;
	}
	
	/**
	 * If operation is executed in MR, only certain operations allow
	 * all formats. In general, unary operations also allow for cell inputs. 
	 * TODO: check and test current format assumptions
	 * 
	 * @param node
	 * @return
	 */
	public boolean checkValidFormatInMR( GDFNode node )
	{
		boolean ret = true;
		
		if( _conf.getExecType()==ExecType.MR ) 
		{
			if( _childs != null )
				for( Plan c : _childs )
					ret &= node.isValidInputFormatForOperation(c._conf.getFormat());
		}
		
		return ret;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("PLAN [");
		sb.append(_ips.toString());
		sb.append(",");
		sb.append(_conf.toString());
		sb.append(",");
		//TODO childs
		sb.append(_costs);
		sb.append("]");
		
		return sb.toString();
	}
}
