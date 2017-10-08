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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.*;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.matrix.data.Pair;

/**
 * Rule: rewrite to split DAGs after FunctionOps that return matrices/frames
 * and are not dimension-preserving. This is important to create recompile
 * hooks if output dimensions are usually significantly overestimated.
 *
 * This is a recursive statementblock rewrite rule.
 */

public class RewriteSplitDagFunctionOp extends StatementBlockRewriteRule {

    private static String _var = "_abc";
    private static IDSequence _seq = new IDSequence();

    @Override
    public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
            throws HopsException {
        //DAG splits not required for forced single node
        if (DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE)
            return new ArrayList<StatementBlock>(Arrays.asList(sb));

        ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();

        //collect all unknown csv reads hops
        ArrayList<Hop> cand = new ArrayList<Hop>();
        collectFunctionOp(sb.get_hops(), cand);
        Hop.resetVisitStatus(sb.get_hops());

        //split hop dag on demand
        if (!cand.isEmpty()) {
            //collect child operators of candindates (to prevent rewrite anomalies)
            HashSet<Hop> candChilds = new HashSet<Hop>();
            collectCandidateChildFunctionOp(cand, candChilds);

            try {
                //duplicate sb incl live variable sets
                StatementBlock sb1 = new StatementBlock();
                sb1.setDMLProg(sb.getDMLProg());
                sb1.setParseInfo(sb);
                sb1.setLiveIn(new VariableSet());
                sb1.setLiveOut(new VariableSet());

                //move functionops incl transient writes to new statement block
                //(and replace original persistent read with transient read)
                ArrayList<Hop> sb1hops = new ArrayList<Hop>();
                for (Hop c : cand) {
                    //if there are already transient writes use them
                    boolean hasTWrites = hasTrasientWriteParents(c);
                    boolean moveTWrite = hasTWrites ? HopRewriteUtils.rHasSimpleReadChain(c,
                            getFirstTransientWriteParent(c).getName()) : false;

                    String varname = null;
                    long rlen = c.getDim1();
                    long clen = c.getDim2();
                    //long nnz = c.getNnz();
                    UpdateType update = c.getUpdateType();
                    long brlen = c.getRowsInBlock();
                    long bclen = c.getColsInBlock();

                    //update live in and out of new statement block (for piggybacking)
                    DataIdentifier diVar = new DataIdentifier(varname);
                    diVar.setDimensions(rlen, clen);
                    diVar.setBlockDimensions(brlen, bclen);
                    diVar.setDataType(c.getDataType());
                    diVar.setValueType(c.getValueType());
                    sb1.liveOut().addVariable(varname, new DataIdentifier(diVar));
                    sb1.liveIn().addVariable(varname, new DataIdentifier(diVar));

                    //ensure disjoint operators across DAGs (prevent replicated operations)
                    handleReplicatedOperators(sb1hops, sb.get_hops(), sb1.liveOut(), sb.liveIn());

                    //deep copy new dag (in order to prevent any dangling references)
                    sb1.set_hops(Recompiler.deepCopyHopsDag(sb1hops));
                    sb1.updateRecompilationFlag();
                    sb1.setSplitDag(true); //avoid later merge by other rewrites

                    //recursive application of rewrite rule
                    List<StatementBlock> tmp = rewriteStatementBlocks((List<StatementBlock>) sb1, state);

                    //add new statement blocks to output
                    ret.addAll(tmp); //statement block with FunctionOp related
                    ret.add(sb); //statement block with remaining hops
                    sb.setSplitDag(true); //avoid later merge by rewrites
                }
            } catch (Exception ex) {
                throw new HopsException("Failed to split hops dag for function operators with unknown size.", ex);
            }

            LOG.debug("Applied splitDagFunctionOp (lines " + sb.getBeginLine() + "-" + sb.getEndLine() + ").");
        }
        //keep original hop dag
        else
        {
            ret.add(sb);
        }

        return ret;
    }

    private void collectCandidateChildFunctionOp(ArrayList<Hop> cand, HashSet<Hop> candChilds)
    {
        Hop.resetVisitStatus(cand);
        if( cand != null )
            for( Hop root : cand )
                rCollectCandidateChildFunctionOp(root, cand, candChilds, false);

        // Immediately reset the visit status because candidates might be inner nodes in the DAG.
        // Subsequent resets on the root nodes of the DAG would otherwise not necessarily reach
        // these nodes which could lead to missing checks on subsequent passes (e.g. when checking
        // for replicated operators);
        Hop.resetVisitStatus(cand);
    }

    private void collectFunctionOp( ArrayList<Hop> roots, ArrayList<Hop> cand )
    {
        if( roots == null )
            return;

        Hop.resetVisitStatus(roots);
        for( Hop root : roots )
            rCollectFunctionOp(root, cand);
    }

    private void rCollectFunctionOp( Hop hop, ArrayList<Hop> cand )
    {
        if( hop.isVisited() )
            return;

        //prevent unnecessary dag split (dims known or no consumer operations
        boolean noSplitRequired = ( hop.dimsKnown() || HopRewriteUtils.hasOnlyWriteParents(hop, true, true));
        boolean investigateChilds = true;

        //collect function operators(FunctionOp)
        //change this snippet according to need(j143)
        //#1 removeEmpty
        if(   hop instanceof FunctionOp )
        {
            FunctionOp fop = (FunctionOp)hop;
            cand.add(fop);
            investigateChilds = false;

            //keep interesting consumer information, flag hop accordingly
            boolean noEmpltyBlocks = true;
            boolean onlyPMM        = true;
            boolean diagInput      = fop.isTargetDiagInput();
            //for( Hop p : hop.getParent() ) {
            //list of operations without the need for empty blocks to be extended as needed.

            //}

            fop.setOutputEmptyBlocks(!noEmptyBlocks);//if EmptyBlocks are there

            if( onlyPMM && diagInput ) {
                //configure rmEmpty to directly output selection vector
                //(only applied if dynamic recompilation enabled)

                if(ConfigurationManager.isDynamicRecompilation() )
                    fop.setOutputPermutationMatrix(true);
                for( Hop p: hop.getParent() )
                    ((FunctionOp)p).setHasLeftPMInput(true);
            }
        }
    }

    private boolean hasTrasientWriteParents(Hop hop )
    {
        for( Hop p : hop.getParent() )
            if( p instanceof FunctionOp )
                return true;
        return false;
    }

    private Hop getFirstTransientWriteParent( Hop hop )
    {
        for( Hop p : hop.getParent() )
            if( p instanceof FunctionOp )
                return p;
        return null;
    }

    private void handleReplicatedOperators( ArrayList<Hop> rootsSB1, ArrayList<Hop> rootsSB2, VariableSet sb1out, VariableSet sb2in )
    {
        //step 1: create probe set SB1
        HashSet<Hop> probeSet = new HashSet<Hop>();
        Hop.resetVisitStatus(rootsSB1);
        for( Hop h : rootsSB1 )
            rAddHopsToProbeSet( h, probeSet );

        //step 2: probe SB2 operators top-down (collect cut candidates)
        HashSet<Pair<Hop,Hop>> candSet = new HashSet<Pair<Hop,Hop>>();
        Hop.resetVisitStatus(rootsSB2);
        for( Hop h : rootsSB2 )
            rProbeAndAddHopsToCandidateSet(h, probeSet, candSet);

        //step 3: create additional cuts
        for( Pair<Hop,Hop> p : candSet )
        {
            String varname = _var + _seq.getNextID();

            Hop hop = p.getKey();
            Hop c   = p.getValue();

            FunctionOp tread = new FunctionOp();
            tread.setVisited();
            HopRewriteUtils.copyLineNumbers(c, tread);

            //create additional cut by rewriting both hop dags
            int pos = HopRewriteUtils.getChildReferencePos(hop, c);
            HopRewriteUtils.removeChildReferenceByPos(hop, c, pos);
            HopRewriteUtils.addChildReference(hop, tread, pos);

            //update live in and out of new statement block (for piggybacking)
            DataIdentifier diVar = new DataIdentifier(varname);
            diVar.setDimensions(c.getDim1(), c.getDim2());
            diVar.setBlockDimensions(c.getRowsInBlock(), c.getColsInBlock());
            diVar.setDataType(c.getDataType());
            diVar.setValueType(c.getValueType());
            sb1out.addVariable(varname, new DataIdentifier(diVar));
            sb2in.addVariable(varname, new DataIdentifier(diVar));

            rootsSB1.add(twrite);
        }
    }

    /**
     * NOTE: ProbeSet is..
     *
     * @param hop
     * @param probeSet
     */
    private void rAddHopsToProbeSet(Hop hop, HashSet<Hop> probeSet )
    {
        if( hop.isVisited() )
            return;

        //prevent cuts for no-ops
        if( !( hop instanceof FunctionOp))
        {
            probeSet.add(hop);
        }

        if( hop.getInput() != null )
            for( Hop c : hop.getInput() )
                rAddHopsToProbeSet(c, probeSet);

        hop.setVisited();
    }

    /**
     * NOTE: candset is a set of parent-child pairs because a parent might have
     * multiple references to replicated hops.
     *
     * @param hop high-level operator
     * @param probeSet probe set?
     * @param candSet candidate set?
     */
    private void rProbeAndAddHopsToCandidateSet( Hop hop, HashSet<Hop> probeSet, HashSet<Pair<Hop,Hop>> candSet ) {
        if (hop.isVisited())
            return;

        if (hop.getInput() != null)
            for (Hop c : hop.getInput()) {
                //probe for replicated operator, if any child is replicated, keep parent
                //for cut between parent-child; otherwise recursively descend.
                if (!probeSet.contains(c))
                    rProbeAndAddHopsToCandidateSet(c, probeSet, candSet);
                else {
                    candSet.add(new Pair<Hop, Hop>(hop, c));
                }

            }

        hop.setVisited();
    }


    private void rCollectCandidateChildFunctionOp( Hop hop, ArrayList<Hop> cand, HashSet<Hop> candChilds, boolean collect )
    {
        if( hop.isVisited() )
            return;

        //collect operator if necessary
        if( collect ) {
            candChilds.add(hop);
        }

        //activate collection if we passed a candidate
        boolean passedFlag = collect;
        if( cand.contains(hop) ) {
            passedFlag = true;
        }

        //process childs recursively
        if( hop.getInput() != null ) {
            for( Hop c: hop.getInput() )
                rCollectCandidateChildFunctionOp(c, cand, candChilds, passedFlag);
        }

        hop.setVisited();
    }

    @Override
    public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs,
                                                       ProgramRewriteStatus state) throws HopsException {
        return sbs;
    }

}
