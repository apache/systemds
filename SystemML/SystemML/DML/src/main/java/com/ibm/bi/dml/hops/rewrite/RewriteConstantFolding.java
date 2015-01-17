/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.io.IOException;
import java.util.ArrayList;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;

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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TMP_VARNAME = "__cf_tmp";
	
	//reuse basic execution runtime
	private static ProgramBlock     _tmpPB = null;
	private static ExecutionContext _tmpEC = null;
	
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
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
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
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
		return rConstantFoldingExpression(hop);
	}
	
	/**
	 * 
	 * @param root
	 * @throws HopsException
	 */
	private Hop rConstantFoldingExpression( Hop root ) 
		throws HopsException
	{
		if( root.getVisited() == VisitStatus.DONE )
			return root;
		
		//recursively process childs (before replacement to allow bottom-recursion)
		//no iterator in order to prevent concurrent modification
		for( int i=0; i<root.getInput().size(); i++ )
		{
			Hop h = root.getInput().get(i);
			rConstantFoldingExpression(h);
		}
		
		//fold binary op if both are literals / unary op if literal
		if(    root.getDataType() == DataType.SCALAR //scalar ouput
			&& ( isApplicableBinaryOp(root) || isApplicableUnaryOp(root) ) )	
		{ 
			LiteralOp literal = null;
			
			//core constant folding via runtime instructions
			try {
				literal = evalScalarOperation(root); 
			}
			catch(Exception ex)
			{
				LOG.error("Failed to execute constant folding instructions. No abort.", ex);
			}
									
			//replace binary operator with folded constant
			if( literal != null ) 
			{
				//reverse replacement in order to keep common subexpression elimination
				int plen = root.getParent().size();
				if( plen > 0 ) //broot is NOT a DAG root
				{
					for( int i=0; i<root.getParent().size(); i++ ) //for all parents
					{
						Hop parent = root.getParent().get(i);
						for( int j=0; j<parent.getInput().size(); j++ )
						{
							Hop child = parent.getInput().get(j);
							if( root == child )
							{
								//replace operator
								parent.getInput().remove(j);
								parent.getInput().add(j, literal);
							}
						}
					}
					root.getParent().clear();	
				}
				else //broot IS a DAG root
				{
					root = literal;
				}
			}		
		}
		
		//mark processed
		root.setVisited( VisitStatus.DONE );
		return root;
	}
	
	/**
	 * In order to (1) prevent unexpected side effects from constant folding and
	 * (2) for simplicity with regard to arbitrary value type combinations,
	 * we use the same compilation and runtime for constant folding as we would 
	 * use for actual instruction execution. 
	 * 
	 * @return
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 */
	private LiteralOp evalScalarOperation( Hop bop ) 
		throws LopsException, DMLRuntimeException, DMLUnsupportedOperationException, IOException, HopsException
	{
		//Timing time = new Timing( true );
		
		DataOp tmpWrite = new DataOp(TMP_VARNAME, bop.getDataType(), bop.getValueType(), bop, DataOpTypes.TRANSIENTWRITE, TMP_VARNAME);
		
		//generate runtime instruction
		Dag<Lop> dag = new Dag<Lop>();
		Recompiler.rClearLops(tmpWrite); //prevent lops reuse
		Lop lops = tmpWrite.constructLops(); //reconstruct lops
		lops.addToDag( dag );	
		ArrayList<Instruction> inst = dag.getJobs(null, ConfigurationManager.getConfig());
		
		//execute instructions
		ExecutionContext ec = getExecutionContext();
		ProgramBlock pb = getProgramBlock();
		pb.setInstructions( inst );
		
		pb.execute( ec );
		
		//get scalar result (check before invocation)
		ScalarObject so = (ScalarObject) ec.getVariable(TMP_VARNAME);
		LiteralOp literal = null;
		switch( bop.getValueType() ){
			case DOUBLE:  literal = new LiteralOp(String.valueOf(so.getDoubleValue()),so.getDoubleValue()); break;
			case INT:     literal = new LiteralOp(String.valueOf(so.getLongValue()),so.getLongValue()); break;
			case BOOLEAN: literal = new LiteralOp(String.valueOf(so.getBooleanValue()),so.getBooleanValue()); break;
			case STRING:  literal = new LiteralOp(String.valueOf(so.getStringValue()),so.getStringValue()); break;	
		}
		
		//cleanup
		tmpWrite.getInput().clear();
		bop.getParent().remove(tmpWrite);
		pb.setInstructions(null);
		ec.getVariables().removeAll();
		
		//set literal properties (scalar)
 		literal.setDim1(0);
		literal.setDim2(0);
		literal.setRowsInBlock(-1);
		literal.setColsInBlock(-1);
		
		//System.out.println("Constant folded in "+time.stop()+"ms.");
		
		return literal;
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static ProgramBlock getProgramBlock() 
		throws DMLRuntimeException
	{
		if( _tmpPB == null )
			_tmpPB = new ProgramBlock( new Program() );
		return _tmpPB;
	}
	
	/**
	 * 
	 * @return
	 */
	private static ExecutionContext getExecutionContext()
	{
		if( _tmpEC == null )
			_tmpEC = new ExecutionContext();
		return _tmpEC;
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 */
	private boolean isApplicableBinaryOp( Hop hop )
	{
		ArrayList<Hop> in = hop.getInput();
		return (   hop instanceof BinaryOp 
				&& in.get(0) instanceof LiteralOp 
				&& in.get(1) instanceof LiteralOp
				&& ((BinaryOp)hop).getOp()!=OpOp2.APPEND );
		
		//string append is rejected although possible because it
		//messes up the explain runtime output due to introduced \n 
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 */
	private boolean isApplicableUnaryOp( Hop hop )
	{
		ArrayList<Hop> in = hop.getInput();
		return (   hop instanceof UnaryOp 
				&& in.get(0) instanceof LiteralOp 
				&& HopRewriteUtils.isValueTypeCast(((UnaryOp)hop).get_op()));			
	}
}
