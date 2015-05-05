/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;

/** 
 * Note: this hop should be called AggQuaternaryOp in consistency with AggUnaryOp and AggBinaryOp;
 * however, since there does not exist a real QuaternaryOp yet - we can leave it as is for now. 
 */
public class QuaternaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private OpOp4 _op = null;
	private boolean _postWeights = false;
	
	private QuaternaryOp() {
		//default constructor for clone
	}
	
	public QuaternaryOp(String l, DataType dt, ValueType vt, Hop.OpOp4 o,
			Hop inX, Hop inU, Hop inV, Hop inW, boolean post) {
		super(Hop.Kind.QuaternaryOp, l, dt, vt);
		_op = o;
		getInput().add(0, inX);
		getInput().add(1, inU);
		getInput().add(2, inV);
		getInput().add(3, inW);
		inX.getParent().add(this);
		inU.getParent().add(this);
		inV.getParent().add(this);
		inW.getParent().add(this);
		
		_postWeights = post;
	}
	
	public OpOp4 getOp(){
		return _op;
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{	
		if (getLops() == null) 
		{
			try 
			{
				ExecType et = optFindExecType();
				WeightsType wtype = checkWeightsType();
				
				switch( _op ) {
					case WSLOSS:
						if( et == ExecType.CP )
							constructCPLopsWeightedSquaredLoss(wtype);
						else if( et == ExecType.MR )
							constructMRLopsWeightedSquaredLoss(wtype);
						else
							throw new HopsException("Unsupported quaternaryop exec type: "+et);
						break;
			
					default:
						throw new HopsException(this.printErrorLocation() + "Unknown QuaternaryOp (" + _op + ") while constructing Lops");
				}
			} 
			catch(LopsException e) {
				throw new HopsException(this.printErrorLocation() + "error constructing lops for QuaternaryOp." , e);
			}
		}
	
		return getLops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "q(" + HopsOpOp4String.get(_op) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() 
		throws HopsException 
	{
		throw new HopsException("QuaternaryOp does not support SQL lops yet!");
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}

	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructCPLopsWeightedSquaredLoss(WeightsType wtype) 
		throws HopsException, LopsException
	{
		Lop wsloss = new WeightedSquaredLoss(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getInput().get(3).constructLops(),
				getDataType(), getValueType(), wtype, ExecType.CP);
		
		setOutputDimensions( wsloss );
		setLineNumbers( wsloss );
		setLops( wsloss );
	}
	
	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructMRLopsWeightedSquaredLoss(WeightsType wtype) 
		throws HopsException, LopsException
	{
		/*Lop matmultCP = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
				                 getDataType(), getValueType(), ExecType.CP, mmtsj);
	
		matmultCP.getOutputParameters().setDimensions(getDim1(), getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
		setLineNumbers( matmultCP );
		setLops(matmultCP);*/
	}
	
	/**
	 * 
	 * @return
	 */
	private WeightsType checkWeightsType()
	{
		WeightsType ret = WeightsType.NONE;
		if( !(getInput().get(3) instanceof LiteralOp) ){
			if( _postWeights )
				ret = WeightsType.POST;
			else
				ret = WeightsType.PRE;
		}
		
		return ret;
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		switch( _op ) {
			case WSLOSS: //always scalar output 
				return OptimizerUtils.DOUBLE_SIZE;
			default:
				return 0;
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz ) 
	{
		switch( _op ) {
			case WSLOSS: //no intermediates 
				return 0;
			default:
				return 0;
		}
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
		//MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
		
		switch( _op ) {
			case WSLOSS: //always scalar output
				ret = null;
				break;
				
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
				
		return ret;
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{	
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if ( (getInput().get(0).areDimsBelowThreshold() 
					&& getInput().get(1).areDimsBelowThreshold()
					&& getInput().get(2).areDimsBelowThreshold()
					&& getInput().get(3).areDimsBelowThreshold()) )
				_etype = ExecType.CP;
			else
				_etype = ExecType.MR;
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();

			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		switch( _op ) {
			case WSLOSS: //do nothing
				break;
			default:
				break;
		}	
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		QuaternaryOp ret = new QuaternaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._postWeights = _postWeights;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.QuaternaryOp )
			return false;
		
		QuaternaryOp that2 = (QuaternaryOp)that;
		
		//compare basic inputs and weights (always existing)
		boolean ret = (_op == that2._op
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& getInput().get(2) == that2.getInput().get(2)
				&& getInput().get(3) == that2.getInput().get(3));
		
		//compare parameters
		ret &= _postWeights == that2._postWeights;
		
		return ret;
	}
}
