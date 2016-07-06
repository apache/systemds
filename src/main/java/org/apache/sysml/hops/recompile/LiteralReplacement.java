/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.hops.recompile;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.VisitStatus;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.utils.Statistics;

public class LiteralReplacement 
{
	
	//internal configuration parameters
	private static final long REPLACE_LITERALS_MAX_MATRIX_SIZE = 1000000; //10^6 cells (8MB)
	private static final boolean REPORT_LITERAL_REPLACE_OPS_STATS = true; 	
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	protected static void rReplaceLiterals( Hop hop, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;

		if( hop.getInput() != null )
		{
			//indexed access to allow parent-child modifications
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop c = hop.getInput().get(i);
				Hop lit = null;
				
				//conditional apply of literal replacements
				lit = (lit==null) ? replaceLiteralScalarRead(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralValueTypeCastScalarRead(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralValueTypeCastLiteral(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralDataTypeCastMatrixRead(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralValueTypeCastRightIndexing(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralFullUnaryAggregate(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralFullUnaryAggregateRightIndexing(c, vars) : lit;
				
				//replace hop w/ literal on demand
				if( lit != null )
				{
					//replace hop c by literal, for all parents to prevent (1) missed opportunities
					//because hop c marked as visited, and (2) repeated evaluation of uagg ops
					
					if( c.getParent().size() > 1 ) { //multiple parents
						ArrayList<Hop> parents = new ArrayList<Hop>(c.getParent());
						for( Hop p : parents ) {
							int pos = HopRewriteUtils.getChildReferencePos(p, c);
							HopRewriteUtils.removeChildReferenceByPos(p, c, pos);
							HopRewriteUtils.addChildReference(p, lit, pos);
						}
					}
					else { //current hop is only parent
						HopRewriteUtils.removeChildReferenceByPos(hop, c, i);
						HopRewriteUtils.addChildReference(hop, lit, i);
					}
				}
				//recursively process children
				else 
				{
					rReplaceLiterals(c, vars);	
				}			
			}
		}
		
		hop.setVisited(VisitStatus.DONE);
	}
	

	///////////////////////////////
	// Literal replacement rules
	///////////////////////////////
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 */
	private static LiteralOp replaceLiteralScalarRead(Hop c, LocalVariableMap vars)
	{
		LiteralOp ret = null;
		
		//scalar read - literal replacement
		if( c instanceof DataOp && ((DataOp)c).getDataOpType() != DataOpTypes.PERSISTENTREAD 
			&& c.getDataType()==DataType.SCALAR )
		{
			Data dat = vars.get(c.getName());
			if( dat != null ) //required for selective constant propagation
			{
				ScalarObject sdat = (ScalarObject)dat;
				switch( sdat.getValueType() ) {
					case INT:
						ret = new LiteralOp(sdat.getLongValue());		
						break;
					case DOUBLE:
						ret = new LiteralOp(sdat.getDoubleValue());	
						break;
					case BOOLEAN:
						ret = new LiteralOp(sdat.getBooleanValue());
						break;
					default:	
						//otherwise: do nothing
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 */
	private static LiteralOp replaceLiteralValueTypeCastScalarRead( Hop c, LocalVariableMap vars )
	{
		LiteralOp ret = null;
		
		//as.double/as.integer/as.boolean over scalar read - literal replacement
		if( c instanceof UnaryOp && (((UnaryOp)c).getOp() == OpOp1.CAST_AS_DOUBLE 
			|| ((UnaryOp)c).getOp() == OpOp1.CAST_AS_INT || ((UnaryOp)c).getOp() == OpOp1.CAST_AS_BOOLEAN )	
				&& c.getInput().get(0) instanceof DataOp && c.getDataType()==DataType.SCALAR )
		{
			Data dat = vars.get(c.getInput().get(0).getName());
			if( dat != null ) //required for selective constant propagation
			{
				ScalarObject sdat = (ScalarObject)dat;
				UnaryOp cast = (UnaryOp) c;
				switch( cast.getOp() ) {
					case CAST_AS_INT:
						ret = new LiteralOp(sdat.getLongValue());		
						break;
					case CAST_AS_DOUBLE:
						ret = new LiteralOp(sdat.getDoubleValue());		
						break;						
					case CAST_AS_BOOLEAN:
						ret = new LiteralOp(sdat.getBooleanValue());		
						break;
					default:	
						//otherwise: do nothing
				}
			}	
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static LiteralOp replaceLiteralValueTypeCastLiteral( Hop c, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		LiteralOp ret = null;
		
		//as.double/as.integer/as.boolean over scalar literal (potentially created by other replacement 
		//rewrite in same dag) - literal replacement
		if( c instanceof UnaryOp && (((UnaryOp)c).getOp() == OpOp1.CAST_AS_DOUBLE 
			|| ((UnaryOp)c).getOp() == OpOp1.CAST_AS_INT || ((UnaryOp)c).getOp() == OpOp1.CAST_AS_BOOLEAN )	
				&& c.getInput().get(0) instanceof LiteralOp )
		{
			LiteralOp sdat = (LiteralOp)c.getInput().get(0);
			UnaryOp cast = (UnaryOp) c;
			try
			{
				switch( cast.getOp() ) {
					case CAST_AS_INT:
						long ival = HopRewriteUtils.getIntValue(sdat);
						ret = new LiteralOp(ival);		
						break;
					case CAST_AS_DOUBLE:
						double dval = HopRewriteUtils.getDoubleValue(sdat);
						ret = new LiteralOp(dval);		
						break;						
					case CAST_AS_BOOLEAN:
						boolean bval = HopRewriteUtils.getBooleanValue(sdat);
						ret = new LiteralOp(bval);		
						break;
					default:	
						//otherwise: do nothing
				}
			}
			catch(HopsException ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static LiteralOp replaceLiteralDataTypeCastMatrixRead( Hop c, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		LiteralOp ret = null;
		
		//as.scalar/matrix read - literal replacement
		if( c instanceof UnaryOp && ((UnaryOp)c).getOp() == OpOp1.CAST_AS_SCALAR 
			&& c.getInput().get(0) instanceof DataOp
			&& c.getInput().get(0).getDataType() == DataType.MATRIX )
		{
			Data dat = vars.get(c.getInput().get(0).getName());
			if( dat != null ) //required for selective constant propagation
			{
				//cast as scalar (see VariableCPInstruction)
				MatrixObject mo = (MatrixObject)dat;
				MatrixBlock mBlock = mo.acquireRead();
				if( mBlock.getNumRows()!=1 || mBlock.getNumColumns()!=1 )
					throw new DMLRuntimeException("Dimension mismatch - unable to cast matrix of dimension ("+mBlock.getNumRows()+" x "+mBlock.getNumColumns()+") to scalar.");
				double value = mBlock.getValue(0,0);
				mo.release();
				
				//literal substitution (always double)
				ret = new LiteralOp(value);
			}
		}

		return ret;
	}
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static LiteralOp replaceLiteralValueTypeCastRightIndexing( Hop c, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		LiteralOp ret = null;
		
		//as.scalar/right indexing w/ literals/vars and matrix less than 10^6 cells
		if( c instanceof UnaryOp && ((UnaryOp)c).getOp() == OpOp1.CAST_AS_SCALAR 
			&& c.getInput().get(0) instanceof IndexingOp
			&& c.getInput().get(0).getDataType() == DataType.MATRIX)
		{
			IndexingOp rix = (IndexingOp)c.getInput().get(0);
			Hop data = rix.getInput().get(0);
			Hop rl = rix.getInput().get(1);
			Hop ru = rix.getInput().get(2);
			Hop cl = rix.getInput().get(3);
			Hop cu = rix.getInput().get(4);
			if(    rix.dimsKnown() && rix.getDim1()==1 && rix.getDim2()==1
				&& data instanceof DataOp && vars.keySet().contains(data.getName())
				&& isIntValueDataLiteral(rl, vars) && isIntValueDataLiteral(ru, vars) 
				&& isIntValueDataLiteral(cl, vars) && isIntValueDataLiteral(cu, vars) ) 
			{
				long rlval = getIntValueDataLiteral(rl, vars);
				long clval = getIntValueDataLiteral(cl, vars);

				MatrixObject mo = (MatrixObject)vars.get(data.getName());
				
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				if( mo.getNumRows()*mo.getNumColumns() < REPLACE_LITERALS_MAX_MATRIX_SIZE )
				{
					MatrixBlock mBlock = mo.acquireRead();
					double value = mBlock.getValue((int)rlval-1,(int)clval-1);
					mo.release();
					
					//literal substitution (always double)
					ret = new LiteralOp(value);
				}
			}
		}
		
		return ret;
	}

	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static LiteralOp replaceLiteralFullUnaryAggregate( Hop c, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		LiteralOp ret = null;
		
		//full unary aggregate w/ matrix less than 10^6 cells
		if( c instanceof AggUnaryOp 
			&& isReplaceableUnaryAggregate((AggUnaryOp)c)
			&& c.getInput().get(0) instanceof DataOp
			&& vars.keySet().contains(c.getInput().get(0).getName()) )
		{
			Hop data = c.getInput().get(0);
			MatrixObject mo = (MatrixObject) vars.get(data.getName());
				
			//get the dimension information from the matrix object because the hop
			//dimensions might not have been updated during recompile
			if( mo.getNumRows()*mo.getNumColumns() < REPLACE_LITERALS_MAX_MATRIX_SIZE )
			{
				MatrixBlock mBlock = mo.acquireRead();
				double value = replaceUnaryAggregate((AggUnaryOp)c, mBlock);
				mo.release();
					
				//literal substitution (always double)
				ret = new LiteralOp(value);
			}		
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param c
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static LiteralOp replaceLiteralFullUnaryAggregateRightIndexing( Hop c, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		LiteralOp ret = null;
		
		//full unary aggregate w/ indexed matrix less than 10^6 cells
		if( c instanceof AggUnaryOp 
			&& isReplaceableUnaryAggregate((AggUnaryOp)c)
			&& c.getInput().get(0) instanceof IndexingOp
			&& c.getInput().get(0).getInput().get(0) instanceof DataOp  )
		{
			IndexingOp rix = (IndexingOp)c.getInput().get(0);
			Hop data = rix.getInput().get(0);
			Hop rl = rix.getInput().get(1);
			Hop ru = rix.getInput().get(2);
			Hop cl = rix.getInput().get(3);
			Hop cu = rix.getInput().get(4);
			
			if(    data instanceof DataOp && vars.keySet().contains(data.getName())
				&& isIntValueDataLiteral(rl, vars) && isIntValueDataLiteral(ru, vars) 
				&& isIntValueDataLiteral(cl, vars) && isIntValueDataLiteral(cu, vars)  ) 
			{
				long rlval = getIntValueDataLiteral(rl, vars);
				long ruval = getIntValueDataLiteral(ru, vars);
				long clval = getIntValueDataLiteral(cl, vars);
				long cuval = getIntValueDataLiteral(cu, vars);

				MatrixObject mo = (MatrixObject) vars.get(data.getName());	
				
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				if( mo.getNumRows()*mo.getNumColumns() < REPLACE_LITERALS_MAX_MATRIX_SIZE )
				{
					MatrixBlock mBlock = mo.acquireRead();
					MatrixBlock mBlock2 = mBlock.sliceOperations((int)(rlval-1), (int)(ruval-1), (int)(clval-1), (int)(cuval-1), new MatrixBlock());
					double value = replaceUnaryAggregate((AggUnaryOp)c, mBlock2);
					mo.release();
						
					//literal substitution (always double)
					ret = new LiteralOp(value);
				}
			}		
		}
		
		return ret;
	}

	
	///////////////////////////////
	// Utility functions
	///////////////////////////////

	/**
	 * 
	 * @param h
	 * @param vars
	 * @return
	 */
	private static boolean isIntValueDataLiteral(Hop h, LocalVariableMap vars)
	{
		return (  (h instanceof DataOp && vars.keySet().contains(h.getName())) 
				|| h instanceof LiteralOp
				||(h instanceof UnaryOp && (((UnaryOp)h).getOp()==OpOp1.NROW || ((UnaryOp)h).getOp()==OpOp1.NCOL)
				   && h.getInput().get(0) instanceof DataOp && vars.keySet().contains(h.getInput().get(0).getName())) );
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static long getIntValueDataLiteral(Hop hop, LocalVariableMap vars) 
		throws DMLRuntimeException
	{
		long value = -1;
		
		try 
		{
			if( hop instanceof LiteralOp )
			{
				value = HopRewriteUtils.getIntValue((LiteralOp)hop);
			}
			else if( hop instanceof UnaryOp && ((UnaryOp)hop).getOp()==OpOp1.NROW )
			{
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				MatrixObject mo = (MatrixObject)vars.get(hop.getInput().get(0).getName());
				value = mo.getNumRows();
			}
			else if( hop instanceof UnaryOp && ((UnaryOp)hop).getOp()==OpOp1.NCOL )
			{
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				MatrixObject mo = (MatrixObject)vars.get(hop.getInput().get(0).getName());
				value = mo.getNumColumns();
			}
			else
			{
				ScalarObject sdat = (ScalarObject) vars.get(hop.getName());
				value = sdat.getLongValue();
			}
		}
		catch(HopsException ex)
		{
			throw new DMLRuntimeException("Failed to get int value for literal replacement", ex);
		}
		
		return value;
	}
	
	
	/**
	 * 
	 * @param auop
	 * @return
	 */
	private static boolean isReplaceableUnaryAggregate( AggUnaryOp auop )
	{
		boolean cdir = (auop.getDirection() == Direction.RowCol);		
		boolean cop = (  auop.getOp() == AggOp.SUM
				      || auop.getOp() == AggOp.SUM_SQ
				      || auop.getOp() == AggOp.MIN
				      || auop.getOp() == AggOp.MAX ); 
		
		return cdir && cop;
	}
	
	/**
	 * 
	 * @param auop
	 * @param mb
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static double replaceUnaryAggregate( AggUnaryOp auop, MatrixBlock mb ) 
		throws DMLRuntimeException
	{
		//setup stats reporting if necessary
		boolean REPORT_STATS = (DMLScript.STATISTICS && REPORT_LITERAL_REPLACE_OPS_STATS); 
		long t0 = REPORT_STATS ? System.nanoTime() : 0;
		
		//compute required unary aggregate 
		double val = Double.MAX_VALUE;
		switch( auop.getOp() ) {
			case SUM: 
				val = mb.sum(); 
				break;
			case SUM_SQ:
				val = mb.sumSq();
				break;
			case MIN:
				val = mb.min(); 
				break;
			case MAX: 
				val = mb.max(); 
				break;
			default:
				throw new DMLRuntimeException("Unsupported unary aggregate replacement: "+auop.getOp());
		}

		//report statistics if necessary
		if( REPORT_STATS ){
			long t1 = System.nanoTime();
			Statistics.maintainCPHeavyHitters("rlit", t1-t0);
		}
		
		return val;
	}

}
