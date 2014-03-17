/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.BinaryCP.OperationTypes;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BinaryCPInstruction;
import com.ibm.bi.dml.runtime.matrix.operators.BinaryOperator;

/**
 * Rule: Constant Folding. For all statement blocks, 
 * eliminate simple binary expressions of literals within dags by 
 * computing them and replacing them with a new Literal op once.
 * For the moment, this only applies within a dag, later this should be 
 * extended across statements block (global, inter-procedure). 
 */
public class RewriteConstantFolding extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
		throws HopsException 
	{
		if( roots == null )
			return null;

		for( int i=0; i<roots.size(); i++ )
		{
			Hop h = roots.get(i);
			roots.set(i, rule_ConstantFolding(h));
		}
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException 
	{
		if( root == null )
			return null;

		return rule_ConstantFolding(root);
	}
	

	/**
	 * 
	 * @param hop
	 * @throws HopsException
	 */
	private Hop rule_ConstantFolding( Hop hop ) 
		throws HopsException 
	{
		return rConstantFoldingBinaryExpression(hop);
	}
	
	/**
	 * 
	 * @param root
	 * @throws HopsException
	 */
	private Hop rConstantFoldingBinaryExpression( Hop root ) 
		throws HopsException
	{
		if( root.get_visited() == VISIT_STATUS.DONE )
			return root;
		
		//recursively process childs (before replacement to allow bottom-recursion)
		//no iterator in order to prevent concurrent modification
		for( int i=0; i<root.getInput().size(); i++ )
		{
			Hop h = root.getInput().get(i);
			rConstantFoldingBinaryExpression(h);
		}
		
		//fold binary op if both are literals
		if( root instanceof BinaryOp 
			&& root.getInput().get(0) instanceof LiteralOp && root.getInput().get(1) instanceof LiteralOp )
		{ 
			BinaryOp broot = (BinaryOp) root;
			LiteralOp lit1 = (LiteralOp) root.getInput().get(0);	
			LiteralOp lit2 = (LiteralOp) root.getInput().get(1);
			double ret = Double.MAX_VALUE;
			
			if(   (lit1.get_valueType()==ValueType.DOUBLE || lit1.get_valueType()==ValueType.INT)
			   && (lit2.get_valueType()==ValueType.DOUBLE || lit2.get_valueType()==ValueType.INT)
			   &&  root.get_valueType()==ValueType.DOUBLE || root.get_valueType()==ValueType.INT || root.get_valueType()==ValueType.BOOLEAN ) //disable string
			{
				try
				{
					double lret = lit1.getDoubleValue();
					double rret = lit2.getDoubleValue();
					ret = evalScalarBinaryOperator(broot.getOp(), lret, rret);
				}
				catch( DMLRuntimeException ex )
				{
					LOG.error("Failed to execute constant folding instructions.", ex);
					ret = Double.MAX_VALUE;
				}
			}
			
			if( ret!=Double.MAX_VALUE )
			{
				LiteralOp literal = null;
				if( broot.get_valueType()==ValueType.DOUBLE )
					literal = new LiteralOp(String.valueOf(ret), ret);
				else if( broot.get_valueType()==ValueType.INT )
					literal = new LiteralOp(String.valueOf((long)ret), (long)ret);
				else if( broot.get_valueType()==ValueType.BOOLEAN )
					literal = new LiteralOp(String.valueOf(ret!=0), ret!=0);
				
				//reverse replacement in order to keep common subexpression elimination
				int plen = broot.getParent().size();
				if( plen > 0 ) //broot is NOT a DAG root
				{
					for( int i=0; i<broot.getParent().size(); i++ ) //for all parents
					{
						Hop parent = broot.getParent().get(i);
						for( int j=0; j<parent.getInput().size(); j++ )
						{
							Hop child = parent.getInput().get(j);
							if( broot == child )
							{
								//replace operator
								parent.getInput().remove(j);
								parent.getInput().add(j, literal);
							}
						}
					}
					broot.getParent().clear();	
				}
				else //broot IS a DAG root
				{
					root = literal;
				}
			}		
		}
		
		//mark processed
		root.set_visited( VISIT_STATUS.DONE );
		return root;
	}

	/**
	 * In order to prevent unexpected side effects from constant folding,
	 * we use the same runtime for constant folding as we would use for 
	 * actual instruction execution. 
	 * 
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private double evalScalarBinaryOperator(OpOp2 op, double left, double right) 
		throws DMLRuntimeException
	{
		//NOTE: we cannot just just hop strings since e.g., EQUALS has different opcode in Hops and Lops
		//String bopcode = Hop.HopsOpOp2String.get(op);
		
		//get instruction opcode
		OperationTypes otype = Hop.HopsOpOp2LopsBS.get(op);
		if( otype == null )
			throw new DMLRuntimeException("Unknown binary operator type: "+op);
		String bopcode = BinaryCP.getOpcode(otype);
		
		//execute binary operator
		BinaryOperator bop = BinaryCPInstruction.getBinaryOperator(bopcode);
		double ret = bop.fn.execute(left, right);
		
		return ret;
	}
	
}
