package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.ReblockOp;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;


public class Plan 
{
	private static IDSequence _seqID   = new IDSequence();
	
	private long _ID                   = -1;
	private GDFNode _node              = null;
	
	private InterestingProperties _ips = null;
	private RewriteConfig _conf        = null;
	private ArrayList<Plan> _childs    = null;
	private double _costs = -1;
	
	public Plan(GDFNode node, InterestingProperties ips, RewriteConfig rc, ArrayList<Plan> childs)
	{
		_ID = _seqID.getNextID();
		_node = node;
		_ips = ips;
		_conf = rc;
		if( childs != null && !childs.isEmpty() )
			_childs = childs;
		else
			_childs = new ArrayList<Plan>();
	}
	
	public Plan( Plan p )
	{	
		_ID = _seqID.getNextID();
		_node = p._node;
		_ips = new InterestingProperties(p._ips);
		_conf = new RewriteConfig(p._conf);
		_costs = p._costs;
		
		if( p._childs != null && !p._childs.isEmpty() )
			_childs = new ArrayList<Plan>(p._childs);
		else
			_childs = new ArrayList<Plan>();
	}
	
	public GDFNode getNode()
	{
		return _node;
	}
	
	public void addChild( Plan c )
	{
		_childs.add(c);
	}
	
	public ArrayList<Plan> getChilds()
	{
		return _childs;
	}
	
	public InterestingProperties getInterestingProperties()
	{
		return _ips;
	}
	
	public RewriteConfig getRewriteConfig()
	{
		return _conf;
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
			if( size0 > 0 ) { //-1 compatible with everything
				for( Plan c : _childs )
					ret &= (  c._conf.getBlockSize() == size0
					        ||c._conf.getBlockSize() <= 0 );
			}
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
	public boolean checkValidFormatInMR()
	{
		boolean ret = true;
		
		if( _conf.getExecType()==ExecType.MR ) 
		{
			if( _childs != null )
				for( Plan c : _childs )
					ret &= _node.isValidInputFormatForOperation(c._conf.getFormat());
		}
		
		return ret;
	}
	
	public boolean checkValidExecutionType()
	{
		boolean ret = true;
		
		ret &= !( _node.getHop() instanceof FunctionOp && _conf.getExecType()!=ExecType.CP );
		ret &= !( _node.getHop() instanceof ReblockOp &&  _conf.getExecType()!=ExecType.MR );
		
		return ret;
	}
	
	/**
	 * A plan is defined as preferred if its output interesting properties
	 * match the interesting properties of all its matrix inputs.
	 * 
	 * @return
	 */
	public boolean isPreferredPlan()
	{
		boolean ret = true;
		
		for( Plan c : _childs )
			if( c.getNode().getDataType()==DataType.MATRIX )
				ret &= _ips.equals( c.getInterestingProperties() );
		
		return ret;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("PLAN("+_ID+") [");
		sb.append(_ips.toString());
		sb.append(",");
		sb.append(_conf.toString());
		sb.append(",{");
		for( Plan c : _childs ){
			sb.append(c._ID);
			sb.append(",");
		}
		sb.setLength(sb.length()-1);
		sb.append("},");	
		sb.append(_costs);
		sb.append("]");
		
		return sb.toString();
	}
}
