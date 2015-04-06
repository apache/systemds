/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.GroupedAggregate;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.RepMat;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.ParameterizedBuiltin;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLLops;


/**
 * Defines the HOP for calling an internal function (with custom parameters) from a DML script. 
 * 
 */
public class ParameterizedBuiltinOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static boolean COMPILE_PARALLEL_REMOVEEMPTY = true;
	
	//operator type
	private ParamBuiltinOp _op;
	
	//removeEmpty hints
	private boolean _outputEmptyBlocks = true;
	private boolean _outputPermutationMatrix = false;
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();

	private ParameterizedBuiltinOp() {
		//default constructor for clone
	}

	/**
	 * Creates a new HOP for a function call
	 */
	public ParameterizedBuiltinOp(String l, DataType dt, ValueType vt,
			ParamBuiltinOp op, HashMap<String, Hop> inputParameters) {
		super(Hop.Kind.ParameterizedBuiltinOp, l, dt, vt);

		_op = op;

		int index = 0;
		for( Entry<String,Hop> e : inputParameters.entrySet() ) 
		{
			String s = e.getKey();
			Hop input = e.getValue();
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public HashMap<String, Integer> getParamIndexMap(){
		return _paramIndexMap;
	}
		
	@Override
	public String getOpString() {
		return "" + _op;
	}
	
	public ParamBuiltinOp getOp() {
		return _op;
	}

	public void setOutputEmptyBlocks(boolean flag)
	{
		_outputEmptyBlocks = flag;
	}
	
	public void setOutputPermutationMatrix(boolean flag)
	{
		_outputPermutationMatrix = flag;
	}
	
	public Hop getTargetHop()
	{
		Hop targetHop = getInput().get(_paramIndexMap.get("target"));
		
		return targetHop;
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{		
		if (getLops() == null) {

			// construct lops for all input parameters
			HashMap<String, Lop> inputlops = new HashMap<String, Lop>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputlops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}

			if ( _op == ParamBuiltinOp.CDF || _op == ParamBuiltinOp.INVCDF ) 
			{
				// simply pass the hashmap of parameters to the lop

				// set the lop for the function call
				
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,
						HopsParameterizedBuiltinLops.get(_op), getDataType(),
						getValueType());
				
				pbilop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				
				setLops(pbilop);

				// set the dimesnions for the lop for the output
				getLops().getOutputParameters().setDimensions(getDim1(),
						getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			} 
			else if (_op == ParamBuiltinOp.GROUPEDAGG) 
			{
				ExecType et = optFindExecType();
				
				constructLopsGroupedAggregate(inputlops, et);
			}
			else if( _op == ParamBuiltinOp.RMEMPTY ) 
			{
				ExecType et = optFindExecType();
				et = (et == ExecType.MR && !COMPILE_PARALLEL_REMOVEEMPTY ) ? ExecType.CP_FILE : et;
				
				constructLopsRemoveEmpty(inputlops, et);
			} 
			else if(   _op == ParamBuiltinOp.REPLACE ) 
			{
				ExecType et = optFindExecType();
				
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin(
						et, inputlops,
						HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType());
				
				pbilop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				
				setLops(pbilop);

				// set the dimensions for the lop for the output
				getLops().getOutputParameters().setDimensions(getDim1(),
						getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			} 

		}

		return getLops();
	}
	
	private void constructLopsGroupedAggregate(HashMap<String, Lop> inputlops, ExecType et) 
		throws HopsException, LopsException 
	{
		
		if ( et == ExecType.SPARK )  {
			// throw new HopsException("constructLopsGroupedAggregate for ParameterizedBuiltinOp not implemented for Spark");
			et = ExecType.CP;
		}
		
		if ( et == ExecType.MR ) 
		{
			// construct necessary lops: combineBinary/combineTertiary and
			// groupedAgg

			boolean isWeighted = (_paramIndexMap.get(Statement.GAGG_WEIGHTS) != null);
			if (isWeighted) 
			{
				Lop append = BinaryOp.constructAppendLopChain(
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_GROUPS)),
						getInput().get(_paramIndexMap.get(Statement.GAGG_WEIGHTS)),
						DataType.MATRIX, getValueType(), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)));

				// add the combine lop to parameter list, with a new name "combinedinput"
				inputlops.put(GroupedAggregate.COMBINEDINPUT, append);
				inputlops.remove(Statement.GAGG_TARGET);
				inputlops.remove(Statement.GAGG_GROUPS);
				inputlops.remove(Statement.GAGG_WEIGHTS);

			} 
			else 
			{
				Lop append = BinaryOp.constructAppendLop(
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_GROUPS)), 
						DataType.MATRIX, getValueType(), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)));
				
				// add the combine lop to parameter list, with a new name
				// "combinedinput"
				inputlops.put(GroupedAggregate.COMBINEDINPUT, append);
				inputlops.remove(Statement.GAGG_TARGET);
				inputlops.remove(Statement.GAGG_GROUPS);

			}
			
			int colwise = -1;
			long outputDim1=-1, outputDim2=-1;
			Lop numGroups = inputlops.get(Statement.GAGG_NUM_GROUPS);
			if ( !dimsKnown() && numGroups != null && numGroups instanceof Data && ((Data)numGroups).isLiteral() ) {
				long ngroups = ((Data)numGroups).getLongValue();
				
				Lop input = inputlops.get(GroupedAggregate.COMBINEDINPUT);
				long inDim1 = input.getOutputParameters().getNumRows();
				long inDim2 = input.getOutputParameters().getNumCols();
				if(inDim1 > 0 && inDim2 > 0 ) {
					if ( inDim1 > inDim2 )
						colwise = 1;
					else 
						colwise = 0;
				}
				
				if ( colwise == 1 ) {
					outputDim1 = ngroups;
					outputDim2 = 1;
				}
				else if ( colwise == 0 ) {
					outputDim1 = 1;
					outputDim2 = ngroups;
				}
				
			}
			
			GroupedAggregate grp_agg = new GroupedAggregate(inputlops, getDataType(), getValueType());
			
			// output dimensions are unknown at compilation time
			grp_agg.getOutputParameters().setDimensions(outputDim1, outputDim2, getRowsInBlock(), getColsInBlock(), -1);
			grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			setLops(grp_agg);
			setRequiresReblock(true);

			// construct and set reblock lop as current root lop
			constructAndSetReblockLopIfRequired(et);
		}
		else //CP 
		{
			GroupedAggregate grp_agg = new GroupedAggregate(inputlops,
					getDataType(), getValueType(), et);
			// output dimensions are unknown at compilation time
			grp_agg.getOutputParameters().setDimensions(-1, -1, -1, -1, -1);
			grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			// introduce a reblock lop only if it is NOT single_node execution
			if( et == ExecType.CP ){
				//force blocked output in CP (see below)
				grp_agg.getOutputParameters().setDimensions(-1, 1, 1000, 1000, -1);
			}
			//grouped agg, w/o reblock in CP
			setLops(grp_agg);
		}
	}

	private void constructLopsRemoveEmpty(HashMap<String, Lop> inputlops, ExecType et) 
		throws HopsException, LopsException 
	{
		Hop targetHop = getInput().get(_paramIndexMap.get("target"));
		Hop marginHop = getInput().get(_paramIndexMap.get("margin"));
		
		if ( et == ExecType.SPARK )  {
			// throw new HopsException("constructLopsRemoveEmpty for ParameterizedBuiltinOp not implemented for Spark");
			et = ExecType.CP;
		}
		
		if( et == ExecType.CP || et == ExecType.CP_FILE )
		{
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin( et, inputlops,
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType());
			
			pbilop.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(pbilop);
			setLops(pbilop);
		
			
			/* DISABLED because cross-dag rewrites like MMchainopt cannot deal with
			   matrix multiplications with non-matching dimensions.
			
			if( et == ExecType.CP && isTargetDiagInput() && marginHop instanceof LiteralOp 
					 && ((LiteralOp)marginHop).getStringValue().equals("rows")
					 && _outputPermutationMatrix ) //SPECIAL CASE SELECTION VECTOR
			{
				
				//TODO this special case could be taken into account for memory estimates in order
				// to reduce the estimates for the input diag and subsequent matrix multiply
				
				//get input vector (without materializing diag())
				Hop input = targetHop.getInput().get(0);
				long brlen = input.getRowsInBlock();
				long bclen = input.getColsInBlock();
				MemoTable memo = new MemoTable();
			
				boolean isPPredInput = (input instanceof BinaryOp && ((BinaryOp)input).isPPredOperation());
				
				//step1: compute index vectors
				Hop ppred0 = input;
				if( !isPPredInput ) { //ppred only if required
					ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp("0",0));
					HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
					ppred0.refreshSizeInformation();
					ppred0.computeMemEstimate(memo); //select exec type
					HopRewriteUtils.copyLineNumbers(this, ppred0);
				}
				
				UnaryOp cumsum = new UnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, ppred0); 
				HopRewriteUtils.setOutputBlocksizes(cumsum, brlen, bclen);
				cumsum.refreshSizeInformation(); 
				cumsum.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, cumsum);	
			
				BinaryOp sel = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, ppred0, cumsum);
				HopRewriteUtils.setOutputBlocksizes(sel, brlen, bclen); 
				sel.refreshSizeInformation();
				sel.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, sel);
				Lop loutput = sel.constructLops();
				
				//Step 4: cleanup hops (allow for garbage collection)
				HopRewriteUtils.removeChildReference(ppred0, input);
				
				setLops( loutput );
			}
			else //GENERAL CASE
			{
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin( et, inputlops,
						HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType());
				
				pbilop.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(pbilop);
				setLops(pbilop);
			}
			*/
		}
		//special compile for mr removeEmpty-diag 
		else if( et == ExecType.MR && isTargetDiagInput() && marginHop instanceof LiteralOp 
				 && ((LiteralOp)marginHop).getStringValue().equals("rows") )
 		{
			//get input vector (without materializing diag())
			Hop input = targetHop.getInput().get(0);
			long brlen = input.getRowsInBlock();
			long bclen = input.getColsInBlock();
			MemoTable memo = new MemoTable();
		
			boolean isPPredInput = (input instanceof BinaryOp && ((BinaryOp)input).isPPredOperation());
			
			//step1: compute index vectors
			Hop ppred0 = input;
			if( !isPPredInput ) { //ppred only if required
				ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp("0",0));
				HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
				ppred0.refreshSizeInformation();
				ppred0.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, ppred0);
			}
			
			UnaryOp cumsum = new UnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, ppred0); 
			HopRewriteUtils.setOutputBlocksizes(cumsum, brlen, bclen);
			cumsum.refreshSizeInformation(); 
			cumsum.computeMemEstimate(memo); //select exec type
			HopRewriteUtils.copyLineNumbers(this, cumsum);	
		
			Lop loutput = null;
			double mest = AggBinaryOp.footprintInMapper(input.getDim1(), 1, brlen, bclen, brlen, bclen, brlen, bclen, 1, true);
			double mbudget = OptimizerUtils.getRemoteMemBudgetMap(true);
			if( _outputPermutationMatrix && mest < mbudget ) //SPECIAL CASE: SELECTION VECTOR
			{
				BinaryOp sel = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, ppred0, cumsum);
				HopRewriteUtils.setOutputBlocksizes(sel, brlen, bclen); 
				sel.refreshSizeInformation();
				sel.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, sel);
				
				loutput = sel.constructLops();
			}
			else //GENERAL CASE: GENERAL PERMUTATION MATRIX
			{
				//max ensures non-zero entries and at least one output row
				BinaryOp max = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MAX, cumsum, new LiteralOp("1",1));
				HopRewriteUtils.setOutputBlocksizes(max, brlen, bclen); 
				max.refreshSizeInformation();
				max.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, max);
				
				DataGenOp seq = HopRewriteUtils.createSeqDataGenOp(input);
				seq.setName("tmp4");
				seq.refreshSizeInformation(); 
				seq.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, seq);	
				
				//step 2: compute removeEmpty(rows) output via table, seq guarantees right column dimension
				//note: weights always the input (even if isPPredInput) because input also includes 0s
				TertiaryOp table = new TertiaryOp("tmp5", DataType.MATRIX, ValueType.DOUBLE, OpOp3.CTABLE, max, seq, input);
				HopRewriteUtils.setOutputBlocksizes(table, brlen, bclen);
				table.refreshSizeInformation();
				table.setForcedExecType(ExecType.MR); //force MR 
				HopRewriteUtils.copyLineNumbers(this, table);
				table.setDisjointInputs(true);
				table.setOutputEmptyBlocks(_outputEmptyBlocks);
				loutput = table.constructLops();
				
				HopRewriteUtils.removeChildReference(table, input);	
			}
			
			//Step 4: cleanup hops (allow for garbage collection)
			HopRewriteUtils.removeChildReference(ppred0, input);
			
			setLops( loutput );
		}
		//default mr remove empty
		else if( et == ExecType.MR )
		{
			//TODO additional physical operator if offsets fit in memory
			
			if( !(marginHop instanceof LiteralOp) )
				throw new HopsException("Parameter 'margin' must be a literal argument.");
				
			Hop input = targetHop;
			long rlen = input.getDim1();
			long clen = input.getDim2();
			long brlen = input.getRowsInBlock();
			long bclen = input.getColsInBlock();
			long nnz = input.getNnz();
			boolean rmRows = ((LiteralOp)marginHop).getStringValue().equals("rows");
			
			//construct lops via new partial hop dag and subsequent lops construction 
			//in order to reuse of operator selection decisions
			
			//Step1: compute row/col non-empty indicators 
			BinaryOp ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp("0",0));
			HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
			ppred0.refreshSizeInformation();
			ppred0.setForcedExecType(ExecType.MR); //always MR 
			HopRewriteUtils.copyLineNumbers(this, ppred0);
			
			Hop emptyInd = ppred0;
			if( !((rmRows && clen == 1) || (!rmRows && rlen==1)) ){
				emptyInd = new AggUnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, AggOp.MAX, rmRows?Direction.Row:Direction.Col, ppred0);
				HopRewriteUtils.setOutputBlocksizes(emptyInd, brlen, bclen);
				emptyInd.refreshSizeInformation();
				emptyInd.setForcedExecType(ExecType.MR); //always MR
				HopRewriteUtils.copyLineNumbers(this, emptyInd);
			}
			
			//Step 2: compute row offsets for non-empty rows
			Hop cumsumInput = emptyInd;
			if( !rmRows ){
				cumsumInput = new ReorgOp( "tmp3a", DataType.MATRIX, ValueType.DOUBLE, ReOrgOp.TRANSPOSE, emptyInd );
				HopRewriteUtils.setOutputBlocksizes(cumsumInput, brlen, bclen);
				cumsumInput.refreshSizeInformation();
				cumsumInput.computeMemEstimate(new MemoTable()); //select exec type
				HopRewriteUtils.copyLineNumbers(this, cumsumInput);	
			}
		
			UnaryOp cumsum = new UnaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, cumsumInput); 
			HopRewriteUtils.setOutputBlocksizes(cumsum, brlen, bclen);
			cumsum.refreshSizeInformation(); 
			cumsum.computeMemEstimate(new MemoTable()); //select exec type
			HopRewriteUtils.copyLineNumbers(this, cumsum);	
		
			Hop cumsumOutput = cumsum;
			if( !rmRows ){
				cumsumOutput = new ReorgOp( "tmp3b", DataType.MATRIX, ValueType.DOUBLE, ReOrgOp.TRANSPOSE, cumsum );
				HopRewriteUtils.setOutputBlocksizes(cumsumOutput, brlen, bclen);
				cumsumOutput.refreshSizeInformation();
				cumsumOutput.computeMemEstimate(new MemoTable()); //select exec type
				HopRewriteUtils.copyLineNumbers(this, cumsumOutput);	
			}
			
			Hop maxDim = new AggUnaryOp("tmp4", DataType.SCALAR, ValueType.DOUBLE, AggOp.MAX, Direction.RowCol, cumsumOutput); //alternative: right indexing
			HopRewriteUtils.setOutputBlocksizes(maxDim, brlen, bclen);
			maxDim.refreshSizeInformation();
			maxDim.computeMemEstimate(new MemoTable()); //select exec type
			HopRewriteUtils.copyLineNumbers(this, maxDim);
			
			BinaryOp offsets = new BinaryOp("tmp5", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, cumsumOutput, emptyInd);
			HopRewriteUtils.setOutputBlocksizes(offsets, brlen, bclen);
			offsets.refreshSizeInformation();
			offsets.computeMemEstimate(new MemoTable()); //select exec type
			HopRewriteUtils.copyLineNumbers(this, offsets);	
			
			//Step 3: gather non-empty rows/cols into final results 
			Lop linput = input.constructLops();
			Lop loffset = offsets.constructLops();
			Lop lmaxdim = maxDim.constructLops();
			
			boolean requiresRep =   ((clen>bclen || clen<=0) &&  rmRows) 
					             || ((rlen>brlen || rlen<=0) && !rmRows);
			
			if( requiresRep ) {
				Lop pos = BinaryOp.createOffsetLop(input, rmRows); //ncol of left input (determines num replicates)
				loffset = new RepMat(loffset, pos, rmRows, DataType.MATRIX, ValueType.DOUBLE);
				loffset.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
				setLineNumbers(loffset);
			}
			
			Group group1 = new Group(linput, Group.OperationTypes.Sort, getDataType(), getValueType());
			setLineNumbers(group1);
			group1.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
		
			Group group2 = new Group( loffset, Group.OperationTypes.Sort, getDataType(), getValueType());
			setLineNumbers(group2);
			group2.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
		
			HashMap<String, Lop> inMap = new HashMap<String, Lop>();
			inMap.put("target", group1);
			inMap.put("offset", group2);
			inMap.put("maxdim", lmaxdim);
			inMap.put("margin", inputlops.get("margin"));
			
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin( et, inMap,
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType());			
			pbilop.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(pbilop);
		
			Group group3 = new Group( pbilop, Group.OperationTypes.Sort, getDataType(), getValueType());
			setLineNumbers(group3);
			group3.getOutputParameters().setDimensions(-1, -1, brlen, bclen, -1);
			
			Aggregate finalagg = new Aggregate(group3, Aggregate.OperationTypes.Sum, DataType.MATRIX, getValueType(), ExecType.MR);
			finalagg.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(finalagg);
			
			//Step 4: cleanup hops (allow for garbage collection)
			HopRewriteUtils.removeChildReference(ppred0, input);
			
			setLops(finalagg);
		}
	}
	
	@Override
	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug(" " + _op);
			}

			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{	
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);	
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		//CDF always known because 
		
		long[] ret = null;
	
		Hop input = getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET));	
		MatrixCharacteristics mc = memo.getAllInputStats(input);

		if (   _op == ParamBuiltinOp.GROUPEDAGG ) {
			// Get the number of groups provided as part of aggregate() invocation, whenever available.
			if ( _paramIndexMap.get(Statement.GAGG_NUM_GROUPS) != null ) {
				Hop ngroups = getInput().get(_paramIndexMap.get(Statement.GAGG_NUM_GROUPS));
				if(ngroups != null && ngroups.getKind() == Kind.LiteralOp) {
					try {
						long m = ((LiteralOp)ngroups).getLongValue();
						//System.out.println("ParamBuiltinOp.inferOutputCharacteristics(): m="+m);
						return new long[]{m,1,m};
					} catch (HopsException e) {
						throw new RuntimeException(e);
					}
				}
				/*else {
					System.out.println("WARN: dimensions are not inferred: " + (ngroups == null ? "null" : ngroups.getKind()) );
				}*/
			}
			
			// Output dimensions are completely data dependent. In the worst case, 
			// #groups = #rows in the grouping attribute (e.g., categorical attribute is an ID column, say EmployeeID).
			// In such a case, #rows in the output = #rows in the input. Also, output sparsity is 
			// likely to be 1.0 (e.g., groupedAgg(groups=<a ID column>, fn="count"))
			// get the size of longer dimension
			long m = (mc.getRows() > 1 ? mc.getRows() : mc.getCols()); 
			if ( m > 1 )
			{
				//System.out.println("ParamBuiltinOp.inferOutputCharacteristics(): worstcase m="+m);
				ret = new long[]{m, 1, m};
			}
		}
		else if (   _op == ParamBuiltinOp.RMEMPTY ) 
		{ 
			// similar to groupedagg because in the worst-case ouputsize eq inputsize
			// #nnz is exactly the same as in the input but sparsity can be higher if dimensions.
			// change (denser output).
			if ( mc.dimsKnown() )
				ret= new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()}; 
		}
		else if (   _op == ParamBuiltinOp.REPLACE ) 
		{ 
			// the worst-case estimate from the input directly propagates to the output 
			// #nnz depends on the replacement pattern and value, same as input if non-zero
			if ( mc.dimsKnown() )
			{
				if( isNonZeroReplaceArguments() )
					ret= new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()};
				else
					ret= new long[]{mc.getRows(), mc.getCols(), -1};
			}
		}
		
		return ret;
	}
	
	@Override 
	public boolean allowsAllExecTypes()
	{
		return false;
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
			else if (   _op == ParamBuiltinOp.GROUPEDAGG 
					 && this.getInput().get(0).areDimsBelowThreshold() ) 
			{
				_etype = ExecType.CP;
			}
			else
			{
				_etype = ExecType.MR;
			}
			
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
		switch( _op )
		{
			case CDF:
				//do nothing; CDF is a scalar
				break;
			
			case GROUPEDAGG:  
				//output dimension dim1 is completely data dependent 
				setDim2( 1 );
				break;
			
			case RMEMPTY: 
				//one output dimension dim1 or dim2 is completely data dependent 
				Hop target = getInput().get(_paramIndexMap.get("target"));
				String margin = getInput().get(_paramIndexMap.get("margin")).toString();
				if( margin.equals("rows") )
					setDim2( target.getDim2() );
				else if (margin.equals("cols"))
					setDim1( target.getDim1() );
				setNnz( target.getNnz() );
				break;
			
			case REPLACE: 
				//dimensions are exactly known from input, sparsity might increase/decrease if pattern/replacement 0 
				Hop target2 = getInput().get(_paramIndexMap.get("target"));
				setDim1( target2.getDim1() );
				setDim2( target2.getDim2() );
				if( isNonZeroReplaceArguments() )
					setNnz( target2.getNnz() );
				
				break;	
		}
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException 
	{
		ParameterizedBuiltinOp ret = new ParameterizedBuiltinOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._outputEmptyBlocks = _outputEmptyBlocks;
		ret._outputPermutationMatrix = _outputPermutationMatrix;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.ParameterizedBuiltinOp )
			return false;
		
		ParameterizedBuiltinOp that2 = (ParameterizedBuiltinOp)that;	
		boolean ret = (_op == that2._op
					  && _paramIndexMap!=null && that2._paramIndexMap!=null
					  && _paramIndexMap.size() == that2._paramIndexMap.size()
					  && _outputEmptyBlocks == that2._outputEmptyBlocks
					  && _outputPermutationMatrix == that2._outputPermutationMatrix );
		if( ret )
		{
			for( Entry<String,Integer> e : _paramIndexMap.entrySet() )
			{
				String key1 = e.getKey();
				int pos1 = e.getValue();
				int pos2 = that2._paramIndexMap.get(key1);
				ret &= (   that2.getInput().get(pos2)!=null
					    && getInput().get(pos1) == that2.getInput().get(pos2) );
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	@Override
	public boolean isTransposeSafe()
	{
		boolean ret = false;
		
		try
		{
			if( _op == ParamBuiltinOp.GROUPEDAGG )
			{
				int ix = _paramIndexMap.get(Statement.GAGG_FN);
				Hop fnHop = getInput().get(ix);
				ret = (fnHop instanceof LiteralOp && Statement.GAGG_FN_SUM.equals(((LiteralOp)fnHop).getStringValue()) );
			}
		}
		catch(Exception ex) {
			//silent false
			LOG.warn("Check for transpose-safeness failed, continue assuming false.", ex);
		}
		
		return ret;	
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isCountFunction()
	{
		boolean ret = false;
		
		try
		{
			if( _op == ParamBuiltinOp.GROUPEDAGG )
			{
				int ix = _paramIndexMap.get(Statement.GAGG_FN);
				Hop fnHop = getInput().get(ix);
				ret = (fnHop instanceof LiteralOp && Statement.GAGG_FN_COUNT.equals(((LiteralOp)fnHop).getStringValue()) );
			}
		}
		catch(Exception ex){
			//silent false
			LOG.warn("Check for count function failed, continue assuming false.", ex);
		}
		
		return ret;
	}
	
	/**
	 * Only applies to REPLACE.
	 * @return
	 */
	private boolean isNonZeroReplaceArguments()
	{
		boolean ret = false;
		try 
		{
			Hop pattern = getInput().get(_paramIndexMap.get("pattern"));
			Hop replace = getInput().get(_paramIndexMap.get("replacement"));
			if( pattern instanceof LiteralOp && ((LiteralOp)pattern).getDoubleValue()!=0d &&
			    replace instanceof LiteralOp && ((LiteralOp)replace).getDoubleValue()!=0d )
			{
				ret = true;
			}
		}
		catch(Exception ex) 
		{
			LOG.warn(ex.getMessage());	
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isTargetDiagInput()
	{
		Hop targetHop = getTargetHop();
		
		//input vector (guarantees diagV2M), implies remove rows
		return (   targetHop instanceof ReorgOp 
				&& ((ReorgOp)targetHop).getOp()==ReOrgOp.DIAG 
				&& targetHop.getInput().get(0).getDim2() == 1 ); 
	}
	
}
