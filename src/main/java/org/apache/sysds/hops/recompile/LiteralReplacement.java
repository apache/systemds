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

package org.apache.sysds.hops.recompile;

import java.util.ArrayList;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.utils.Statistics;

public class LiteralReplacement 
{
	//internal configuration parameters
	private static final long REPLACE_LITERALS_MAX_MATRIX_SIZE = 1000000; //10^6 cells (8MB)
	private static final boolean REPORT_LITERAL_REPLACE_OPS_STATS = true;
	
	protected static void rReplaceLiterals( Hop hop, ExecutionContext ec, boolean scalarsOnly )
	{
		if( hop.isVisited() )
			return;

		if( hop.getInput() != null )
		{
			//indexed access to allow parent-child modifications
			LocalVariableMap vars = ec.getVariables();
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop c = hop.getInput().get(i);
				Hop lit = null;
				
				//conditional apply of literal replacements
				lit = replaceLiteralScalarRead(c, vars);
				lit = (lit==null) ? replaceLiteralValueTypeCastScalarRead(c, vars) : lit;
				lit = (lit==null) ? replaceLiteralValueTypeCastLiteral(c, vars) : lit;
				if( !scalarsOnly ) {
					lit = (lit==null) ? replaceLiteralDataTypeCastMatrixRead(c, vars) : lit;
					lit = (lit==null) ? replaceLiteralValueTypeCastRightIndexing(c, vars) : lit;
					lit = (lit==null) ? replaceLiteralFullUnaryAggregate(c, vars) : lit;
					lit = (lit==null) ? replaceLiteralFullUnaryAggregateRightIndexing(c, vars) : lit;
					lit = (lit==null) ? replaceTReadMatrixFromList(c, ec) : lit;
					lit = (lit==null) ? replaceTReadMatrixFromListAppend(c, ec) : lit;
					lit = (lit==null) ? replaceTReadMatrixLookupFromList(c, vars) : lit;
					lit = (lit==null) ? replaceTReadScalarLookupFromList(c, vars) : lit;
				}
				
				//replace hop w/ literal on demand
				if( lit != null )
				{
					//replace hop c by literal, for all parents to prevent (1) missed opportunities
					//because hop c marked as visited, and (2) repeated evaluation of uagg ops
					
					if( c.getParent().size() > 1 ) { //multiple parents
						ArrayList<Hop> parents = new ArrayList<>(c.getParent());
						for( Hop p : parents ) {
							int pos = HopRewriteUtils.getChildReferencePos(p, c);
							HopRewriteUtils.removeChildReferenceByPos(p, c, pos);
							HopRewriteUtils.addChildReference(p, lit, pos);
						}
					}
					else { //current hop is only parent
						HopRewriteUtils.replaceChildReference(hop, c, lit, i);
					}
				}
				//recursively process children
				else {
					rReplaceLiterals(c, ec, scalarsOnly);
				}
			}
		}
		
		hop.setVisited();
	}
	
	///////////////////////////////
	// Literal replacement rules
	///////////////////////////////
	
	private static LiteralOp replaceLiteralScalarRead(Hop c, LocalVariableMap vars)
	{
		LiteralOp ret = null;
		
		//scalar read - literal replacement
		if( c instanceof DataOp && ((DataOp)c).getOp() != OpOpData.PERSISTENTREAD 
			&& c.getDataType()==DataType.SCALAR )
		{
			Data dat = vars.get(c.getName());
			if( dat != null ) { //required for selective constant propagation
				ScalarObject sdat = (ScalarObject)dat;
				ret = ScalarObjectFactory.createLiteralOp(sdat);
			}
		}
		
		return ret;
	}
	
	private static LiteralOp replaceLiteralValueTypeCastScalarRead(Hop c, LocalVariableMap vars)
	{
		LiteralOp ret = null;
		
		//as.double/as.integer/as.boolean over scalar read - literal replacement
		if( c instanceof UnaryOp && (((UnaryOp)c).getOp() == OpOp1.CAST_AS_DOUBLE 
			|| ((UnaryOp)c).getOp() == OpOp1.CAST_AS_INT || ((UnaryOp)c).getOp() == OpOp1.CAST_AS_BOOLEAN )	
				&& c.getInput().get(0) instanceof DataOp && c.getDataType()==DataType.SCALAR )
		{
			Data dat = vars.get(c.getInput().get(0).getName());
			if( dat != null ) { //required for selective constant propagation
				ScalarObject sdat = (ScalarObject)dat;
				ret = ScalarObjectFactory.createLiteralOp(sdat, (UnaryOp) c);
			}
		}
		
		return ret;
	}
	
	private static LiteralOp replaceLiteralValueTypeCastLiteral(Hop c, LocalVariableMap vars)
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
	
	private static LiteralOp replaceLiteralDataTypeCastMatrixRead( Hop c, LocalVariableMap vars )
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
	
	private static LiteralOp replaceLiteralValueTypeCastRightIndexing( Hop c, LocalVariableMap vars ) 
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

	private static LiteralOp replaceLiteralFullUnaryAggregate( Hop c, LocalVariableMap vars )
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
	
	private static LiteralOp replaceLiteralFullUnaryAggregateRightIndexing( Hop c, LocalVariableMap vars )
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
					MatrixBlock mBlock2 = mBlock.slice((int)(rlval-1), (int)(ruval-1), (int)(clval-1), (int)(cuval-1), new MatrixBlock());
					double value = replaceUnaryAggregate((AggUnaryOp)c, mBlock2);
					mo.release();
						
					//literal substitution (always double)
					ret = new LiteralOp(value);
				}
			}		
		}
		
		return ret;
	}
	
	private static DataOp replaceTReadMatrixFromList( Hop c, ExecutionContext ec ) {
		//pattern: as.matrix(X) or as.matrix(X) with X being a list
		DataOp ret = null;
		if( HopRewriteUtils.isUnary(c, OpOp1.CAST_AS_MATRIX) ) {
			Hop in = c.getInput().get(0);
			if( in.getDataType() == DataType.LIST
				&& HopRewriteUtils.isData(in, OpOpData.TRANSIENTREAD) ) {
				ListObject list = (ListObject)ec.getVariables().get(in.getName());
				if( list.getLength() == 1 ) {
					String varname = Dag.getNextUniqueVarname(DataType.MATRIX);
					MatrixObject mo = (MatrixObject) list.slice(0);
					ec.getVariables().put(varname, mo);
					ret = HopRewriteUtils.createTransientRead(varname, c);
					if (DMLScript.LINEAGE)
						ec.getLineage().set(varname, list.getLineageItem(0));
				}
			}
		}
		return ret;
	}
	
	private static NaryOp replaceTReadMatrixFromListAppend( Hop c, ExecutionContext ec ) {
		//pattern: cbind(X) or rbind(X) with X being a list
		NaryOp ret = null;
		if( HopRewriteUtils.isNary(c, OpOpN.CBIND, OpOpN.RBIND)) {
			Hop in = c.getInput().get(0);
			if( in.getDataType() == DataType.LIST
				&& HopRewriteUtils.isData(in, OpOpData.TRANSIENTREAD) ) {
				ListObject list = (ListObject)ec.getVariables().get(in.getName());
				if( list.getLength() <= 128 ) {
					ArrayList<Hop> tmp = new ArrayList<>();
					for( int i=0; i < list.getLength(); i++ ) {
						String varname = Dag.getNextUniqueVarname(DataType.MATRIX);
						MatrixObject mo = (MatrixObject) list.slice(i);
						ec.getVariables().put(varname, mo);
						tmp.add(HopRewriteUtils.createTransientRead(varname, mo));
						if (DMLScript.LINEAGE)
							ec.getLineage().set(varname, list.getLineageItem(i));
					}
					ret = HopRewriteUtils.createNary(
						((NaryOp)c).getOp(), tmp.toArray(new Hop[0]));
				}
			}
		}
		return ret;
	}

	private static DataOp replaceTReadMatrixLookupFromList( Hop c, LocalVariableMap vars ) {
		//pattern: as.matrix(X[i:i]) or as.matrix(X['a','a']) with X being a list
		DataOp ret = null;
		if( HopRewriteUtils.isUnary(c, OpOp1.CAST_AS_MATRIX)
			&& c.getInput().get(0) instanceof IndexingOp ) {
			Hop ix = c.getInput().get(0);
			Hop ixIn = c.getInput().get(0).getInput().get(0);
			if( ixIn.getDataType() == DataType.LIST
				&& HopRewriteUtils.isData(ixIn, OpOpData.TRANSIENTREAD)
				&& ix.getInput().get(1) instanceof LiteralOp 
				&& ix.getInput().get(2) instanceof LiteralOp
				&& ix.getInput().get(1) == ix.getInput().get(2) ) {
				ListObject list = (ListObject)vars.get(ixIn.getName());
				String varname = Dag.getNextUniqueVarname(DataType.MATRIX);
				LiteralOp lit = (LiteralOp) ix.getInput().get(1);
				MatrixObject mo = (MatrixObject) (!lit.getValueType().isNumeric() ?
					list.slice(lit.getName()) : list.slice((int)lit.getLongValue()-1));
				vars.put(varname, mo);
				ret = HopRewriteUtils.createTransientRead(varname, c);
			}
		}
		return ret;
	}
	
	private static LiteralOp replaceTReadScalarLookupFromList( Hop c, LocalVariableMap vars ) {
		//pattern: as.scalar(X[i:i]) or as.scalar(X['a','a']) with X being a list
		if( HopRewriteUtils.isUnary(c, OpOp1.CAST_AS_SCALAR)
			&& c.getInput().get(0) instanceof IndexingOp ) {
			Hop ix = c.getInput().get(0);
			Hop ixIn = c.getInput().get(0).getInput().get(0);
			if( ixIn.getDataType() == DataType.LIST
				&& HopRewriteUtils.isData(ixIn, OpOpData.TRANSIENTREAD)
				&& ix.getInput().get(1) instanceof LiteralOp 
				&& ix.getInput().get(2) instanceof LiteralOp
				&& ix.getInput().get(1) == ix.getInput().get(2) ) {
				ListObject list = (ListObject)vars.get(ixIn.getName());
				LiteralOp lit = (LiteralOp) ix.getInput().get(1);
				ScalarObject so = (ScalarObject) (!lit.getValueType().isNumeric() ?
					list.slice(lit.getName()) : list.slice((int)lit.getLongValue()-1));
				return ScalarObjectFactory.createLiteralOp(so);
			}
		}
		return null;
	}
	
	///////////////////////////////
	// Utility functions
	///////////////////////////////

	private static boolean isIntValueDataLiteral(Hop h, LocalVariableMap vars)
	{
		return ( (h instanceof DataOp && vars.keySet().contains(h.getName())) 
			|| h instanceof LiteralOp
			||(h instanceof UnaryOp && (((UnaryOp)h).getOp()==OpOp1.NROW || ((UnaryOp)h).getOp()==OpOp1.NCOL)
				&& h.getInput().get(0) instanceof DataOp && vars.keySet().contains(h.getInput().get(0).getName())) );
	}
	
	private static long getIntValueDataLiteral(Hop hop, LocalVariableMap vars)
	{
		long value = -1;
		
		try 
		{
			if( hop instanceof LiteralOp ) {
				value = HopRewriteUtils.getIntValue((LiteralOp)hop);
			}
			else if( hop instanceof UnaryOp && ((UnaryOp)hop).getOp()==OpOp1.NROW ) {
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				CacheableData<?> mo = (CacheableData<?>)vars.get(hop.getInput().get(0).getName());
				value = mo.getNumRows();
			}
			else if( hop instanceof UnaryOp && ((UnaryOp)hop).getOp()==OpOp1.NCOL ) {
				//get the dimension information from the matrix object because the hop
				//dimensions might not have been updated during recompile
				CacheableData<?> mo = (CacheableData<?>)vars.get(hop.getInput().get(0).getName());
				value = mo.getNumColumns();
			}
			else {
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
	
	private static boolean isReplaceableUnaryAggregate( AggUnaryOp auop ) {
		boolean cdir = (auop.getDirection() == Direction.RowCol);
		boolean cop = (auop.getOp() == AggOp.SUM
			|| auop.getOp() == AggOp.SUM_SQ
			|| auop.getOp() == AggOp.MIN
			|| auop.getOp() == AggOp.MAX);
		boolean matrixInput = auop.getInput().get(0).getDataType().isMatrix();
		return cdir && cop && matrixInput;
	}
	
	private static double replaceUnaryAggregate( AggUnaryOp auop, MatrixBlock mb )
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
