/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfgraph;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.AggUnaryOp;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.Direction;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;

/**
 * The reason of a custom graph structure is to unify both within DAG
 * and cross DAG enumeration. Conceptually, we would only need interesting
 * properties of transient reads and could compile locally. 
 * 
 * Furthermore, having a global graph structure also allows for more advanced
 * algebraic simplification rewrites because the semantics of transient read
 * inputs are always available.
 * 
 */
public class GDFNode 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum NodeType{
		HOP_NODE,
		LOOP_NODE,
		CROSS_BLOCK_NODE,
	}
	
	private static IDSequence _seqID = new IDSequence();
	
	protected NodeType _type = null;
	protected long _ID = -1;
	
	//references to original program and hop dag
	protected Hop _hop = null;
	protected ProgramBlock _pb = null;
	
	//input nodes
	protected ArrayList<GDFNode> _inputs = null;
	
	public GDFNode()
	{
		_ID = _seqID.getNextID();
	}
	
	public GDFNode( Hop hop, ProgramBlock pb, ArrayList<GDFNode> inputs )
	{
		this();
		_type = NodeType.HOP_NODE;
		_hop = hop;
		_pb = pb;
		_inputs = inputs;
	}
	
	public NodeType getNodeType()
	{
		return _type;
	}
	
	public long getID()
	{
		return _ID;
	}
	
	public Hop getHop()
	{
		return _hop;
	}
	
	public ProgramBlock getProgramBlock()
	{
		return _pb;
	}
	
	public Program getProgram()
	{
		if( _pb != null )
			return _pb.getProgram();
		return null;
	}
	
	public ArrayList<GDFNode> getInputs()
	{
		return _inputs;
	}
	
	public DataType getDataType()
	{
		return _hop.getDataType();
	}
	
	/**
	 * If the output or any input is a matrix we need to consider
	 * MR configurations. This for examples excludes Literals or
	 * any purely scalar operation.
	 * 
	 * @return
	 */
	public boolean requiresMREnumeration()
	{
		//general rule: MR generation required if at least one matrix input/output
		boolean ret = (_hop.getDataType() == DataType.MATRIX);		
		for( Hop c : _hop.getInput() )
			ret |= (c.getDataType() == DataType.MATRIX);
		
		//special cases of CP-only operators
		if( _hop instanceof UnaryOp && ((UnaryOp)_hop).getOp()==OpOp1.CAST_AS_SCALAR ) //as.scalar
			ret = false;
		if( _hop instanceof DataGenOp && ((DataGenOp)_hop).getOp()==DataGenMethod.SINIT ) //matrix(str, )
			ret = false;
		if( _hop instanceof UnaryOp && ((UnaryOp)_hop).getOp()==OpOp1.NROW ) //nrow - meta data only
			ret = false;
		if( _hop instanceof UnaryOp && ((UnaryOp)_hop).getOp()==OpOp1.NCOL ) //ncol - meta data only
			ret = false;
		
		return ret;
	}
	
	/**
	 * 
	 * @param format
	 * @return
	 */
	public boolean isValidInputFormatForOperation( FileFormatTypes format )
	{
		return (   _hop instanceof UnaryOp && format!=FileFormatTypes.CSV
				|| (_hop instanceof AggUnaryOp && ((AggUnaryOp)_hop).getDirection()==Direction.RowCol && format!=FileFormatTypes.CSV)
				|| (_hop instanceof ReorgOp && ((ReorgOp)_hop).getOp()==ReOrgOp.TRANSPOSE && format!=FileFormatTypes.CSV)
				|| format==FileFormatTypes.BINARY ); //any op
	}
	
	/**
	 * 
	 * @param deps
	 * @return
	 */
	public String explain(String deps) 
	{
		String ldeps = (deps!=null) ? deps : "";
		
		//node details
		if( _hop!=null )
			return "Node "+ldeps+" ["+_hop.getHopID()+", "+_hop.getOpString()+"]";
		else
			return "Node "+ldeps+" [null]";
	}
}
