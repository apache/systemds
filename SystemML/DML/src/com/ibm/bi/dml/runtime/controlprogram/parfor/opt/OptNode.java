package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;

import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.Lops;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;

/**
 * Internal representation of a plan alternative for program blocks and instructions 
 * in order to enable efficient and simple recursive enumeration and plan changes.
 * This is only used within the optimizer and therefore not visible to any other component.
 * 
 */
public class OptNode 
{
	public enum NodeType{
		GENERIC,
		IF,
		WHILE,
		FOR,
		PARFOR,
		INST,
		HOP
	}
	
	public enum ExecType{ 
		CP,
		MR;
		
		public LopProperties.ExecType toLopsExecType(){
			return (this == CP)? LopProperties.ExecType.CP : LopProperties.ExecType.MR;
		}
		
		public ParForProgramBlock.PExecMode toParForExecMode(){
			return (this == CP)? ParForProgramBlock.PExecMode.LOCAL : ParForProgramBlock.PExecMode.REMOTE_MR;
		}
	}
	
	public enum ParamType{
		OPTYPE,
		OPSTRING,
		PARTITIONER,
		NUM_ITERATIONS
	}

	//child nodes
	private ArrayList<OptNode>        _childs  = null;

	//node configuration 
	private long                      _id      = -1;
	private NodeType                  _ntype   = null;
	private ExecType                  _etype   = null;
	private int                       _k       = -1;
	private HashMap<ParamType,String> _params  = null;
	
	//node statistics (only present for physical plans and leaf nodes)
	private OptNodeStatistics         _stats   = null;
	
	public OptNode( NodeType type )
	{
		this(type, null);
	}

	public OptNode( NodeType ntype, ExecType etype )
	{
		_ntype = ntype;
		_etype = etype;
		
		_k = 1;
	}
	
	///////
	//getters and setters
	
	public NodeType getNodeType() 
	{
		return _ntype;
	}
	
	public void setNodeType(NodeType type) 
	{
		_ntype = type;
	}
	
	public ExecType getExecType() 
	{
		return _etype;
	}
	
	public void setExecType(ExecType type) 
	{
		_etype = type;
	}
	
	public void setID( long id )
	{
		_id = id;
	}
	
	public long getID( )
	{
		return _id;
	}
	
	public void addParam(ParamType ptype, String val)
	{
		if( _params == null )
			_params = new HashMap<ParamType, String>();
		
		_params.put(ptype, val);
	}

	public void setParams( HashMap<ParamType,String> params )
	{
		_params = params;
	}
	
	public String getParam( ParamType type )
	{
		return _params.get(type);
	}
	
	public void addChild( OptNode child )
	{
		if( _childs==null )
			_childs = new ArrayList<OptNode>();
		
		_childs.add( child );
	}
	
	public void addChilds( ArrayList<OptNode> childs )
	{
		if( _childs==null )
			_childs = new ArrayList<OptNode>();
		
		_childs.addAll( childs );		
	}
	
	public void setChilds(ArrayList<OptNode> childs) 
	{
		_childs = childs;
	}
	
	public ArrayList<OptNode> getChilds() 
	{
		return _childs;
	}
	
	
	public int getK() 
	{
		return _k;
	}

	public void setK(int k) 
	{
		_k = k;
	}
	
	public OptNodeStatistics getStatistics()
	{
		return _stats;
	}
	
	public void setStatistics(OptNodeStatistics stats)
	{
		_stats = stats;
	}
	
	/**
	 * 
	 * @param oldNode
	 * @param newNode
	 * @return
	 */
	public boolean exchangeChild(OptNode oldNode, OptNode newNode) 
	{
		boolean ret = false;
		
		if( _childs != null )
			for( int i=0; i<_childs.size(); i++ )
				if( _childs.get(i) == oldNode )
				{
					_childs.set(i, newNode);
					ret = true;
				}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isLeaf()
	{
		return ( _childs == null || _childs.size()==0 );
	}
	
	/**
	 * 
	 * @return
	 */
	public String getInstructionName() 
	{
		return String.valueOf(_etype) + Lops.OPERAND_DELIMITOR + getParam(ParamType.OPSTRING);
	}
	

	///////
	//recursive methods
	
	
	/**
	 * 
	 * @return
	 */
	public Collection<OptNode> getNodeList()
	{
		Collection<OptNode> nodes = new LinkedList<OptNode>();
		
		if(!isLeaf())
			for( OptNode n : _childs )
				nodes.addAll( n.getNodeList() );
		nodes.add(this);
		
		return nodes;
	}
	
	/**
	 * 
	 * @return
	 */
	public Collection<OptNode> getRelevantNodeList()
	{
		Collection<OptNode> nodes = new LinkedList<OptNode>();
		
		if( !isLeaf() )
		{
			for( OptNode n : _childs )
				nodes.addAll( n.getRelevantNodeList() );
		}
		 
		if( _ntype == NodeType.PARFOR || _ntype == NodeType.HOP )
		{
			nodes.add(this);
		}
		
		return nodes;
	}

	
	/**
	 * Set the plan to a parallel degree of 1 (serial execution).
	 */
	public void setSerialParFor()
	{
		//process parfor nodes
		if( _ntype == NodeType.PARFOR )
		{
			_k = 1;
			_etype = ExecType.CP;
		}
		
		//process childs
		if( _childs != null )
			for( OptNode n : _childs )
				n.setSerialParFor();
	}

	/**
	 * Gets the number of plan nodes.
	 * 
	 * @return
	 */
	public int size() 
	{
		int count = 1; //self
		if( _childs != null )
			for( OptNode n : _childs )
				count += n.size();
		return count;
	}
	
	/**
	 * Determines if all programblocks and instructions exhibit 
	 * the execution type CP. 
	 * 
	 * @return
	 */
	public boolean isCPOnly()
	{
		boolean ret = (_etype == ExecType.CP);		
		if( _childs != null )
			for( OptNode n : _childs )
			{
				if( !ret ) break; //early abort if already false
				ret &= n.isCPOnly();
			}
		return ret;
	}
	
	/**
	 * Explain tool: prints the hierarchical plan to <code>stdout</code>.
	 * 
	 * @param level
	 * @param withDetails
	 * @return
	 */
	public String explain(int level, boolean withDetails) 
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<level; i++ )
			sb.append("--");	
		if( _ntype == NodeType.INST )
			sb.append(_params.get(ParamType.OPSTRING));
		else
			sb.append(_ntype);
		sb.append(", exec=");
		sb.append(_etype);
		sb.append(", k=");
		sb.append(_k);
		sb.append("\n");
		
		if( _childs != null )
			for( OptNode n : _childs )
				sb.append( n.explain(level+1, withDetails) );
		
		return sb.toString();
	}

	/**
	 * Determines the maximum problem size of all childs.
	 * 
	 * @return
	 */
	public int getMaxProblemSize() 
	{
		int max = 0;
		if( _childs != null )
			for( OptNode n : _childs )
				max = Math.max(max, n.getMaxProblemSize());		
		else
			max = 1;
		
		if( _ntype == NodeType.PARFOR )
			max = max * Integer.parseInt(_params.get(ParamType.NUM_ITERATIONS));

		return max;
	}
	
	
	/**
	 * 
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public OptNode createShallowClone()
	{
		OptNode n = new OptNode(_ntype,_etype);
		n.setID(_id);
		n.setK(_k);		
		if( _childs != null )
			n.setChilds( (ArrayList<OptNode>)_childs.clone() );
		if( _params != null )
			n.setParams((HashMap<ParamType,String>)_params.clone());
		return n;
	}
	
	/**
	 * 
	 * @return
	 */
	public OptNode createDeepClone()
	{
		throw new RuntimeException("not implemented yet");
	}


}
