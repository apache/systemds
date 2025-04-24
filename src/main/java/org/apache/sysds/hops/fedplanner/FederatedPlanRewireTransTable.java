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

 package org.apache.sysds.hops.fedplanner;
 import org.apache.commons.lang3.tuple.Pair;
 import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.*;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.parser.*;
import org.apache.sysds.hops.fedplanner.FederatedMemoTable.HopCommon;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import java.util.*;

public class FederatedPlanRewireTransTable {
    private static final double DEFAULT_LOOP_WEIGHT = 10.0;
    private static final double DEFAULT_IF_ELSE_WEIGHT = 0.5;
    
    public static void rewireProgram(DMLProgram prog, Map<Long, List<Hop>> rewireTable, Map<Long, HopCommon> hopCommonTable, 
                                        Set<Long> unRefTwriteSet, Set<Long> unRefSet, Set<Hop> progRootHopSet) {
        // Maps Hop ID and fedOutType pairs to their plan variants
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();
        List<Pair<Long, Double>> loopStack = new ArrayList<>();

        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);

        for (StatementBlock sb : prog.getStatementBlocks()) {
           Map<String, List<Hop>> innerTransTable = rewireStatementBlock(sb, prog, visitedHops, rewireTable, hopCommonTable, outerTransTableList, null, 
                                                                            unRefTwriteSet, unRefSet, progRootHopSet, fnStack, 1, 1, loopStack);
           outerTransTableList.get(0).putAll(innerTransTable);
        }
    }

    public static void rewireFunctionDynamic(FunctionStatementBlock function, Map<Long, List<Hop>> rewireTable,  Map<Long, HopCommon> hopCommonTable, 
                                                Set<Long> unRefTwriteSet, Set<Long> unRefSet, Set<Hop> progRootHopSet) {
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();
        List<Pair<Long, Double>> loopStack = new ArrayList<>();
        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);
        // Todo: not tested
        rewireStatementBlock(function, null, visitedHops, rewireTable, hopCommonTable, outerTransTableList, null, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, 1, 1, loopStack);
    }

    public static Map<String, List<Hop>> rewireStatementBlock(StatementBlock sb, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable, Map<Long, HopCommon> hopCommonTable,
                                                                List<Map<String, List<Hop>>> outerTransTableList, Map<String, List<Hop>> formerTransTable, Set<Long> unRefTwriteSet, Set<Long> unRefSet,
                                                                Set<Hop> progRootHopSet, Set<String> fnStack, double computeWeight, double networkWeight, List<Pair<Long, Double>> parentLoopStack) {
        List<Map<String, List<Hop>>> newOuterTransTableList = new ArrayList<>();                                           
        if (outerTransTableList != null){
            for (Map<String, List<Hop>> outerTable: outerTransTableList){
                if (outerTable != null && !outerTable.isEmpty()){
                    newOuterTransTableList.add(outerTable);
                }
            }
        }
        if (formerTransTable != null && !formerTransTable.isEmpty()){
            newOuterTransTableList.add(formerTransTable);
        }

        Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
        Map<String, List<Hop>> innerTransTable = new HashMap<>();

        if (sb instanceof IfStatementBlock) {
            IfStatementBlock isb = (IfStatementBlock) sb;
            IfStatement istmt = (IfStatement)isb.getStatement(0);

            rewireHopDAG(isb.getPredicateHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null, innerTransTable, 
                            unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, parentLoopStack);

            newFormerTransTable.putAll(innerTransTable);
            Map<String, List<Hop>> elseFormerTransTable = new HashMap<>();
            elseFormerTransTable.putAll(innerTransTable);
            computeWeight *= DEFAULT_IF_ELSE_WEIGHT;

            for (StatementBlock innerIsb : istmt.getIfBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, newFormerTransTable, 
                                                                    unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, parentLoopStack));

            for (StatementBlock innerIsb : istmt.getElseBody())
                elseFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, elseFormerTransTable, 
                                                                     unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, parentLoopStack));

            // If there are common keys: merge elseValue list into ifValue list
            elseFormerTransTable.forEach((key, elseValue) -> {
               newFormerTransTable.merge(key, elseValue, (ifValue, newValue) -> {
                    ifValue.addAll(newValue);
                    return ifValue;
                });
            });
        }
        else if (sb instanceof ForStatementBlock) { //incl parfor
            ForStatementBlock fsb = (ForStatementBlock) sb;
            ForStatement fstmt = (ForStatement)fsb.getStatement(0);

 			 // Calculate for-loop iteration count if possible
              double loopWeight = DEFAULT_LOOP_WEIGHT;
              Hop from = fsb.getFromHops().getInput().get(0);
              Hop to = fsb.getToHops().getInput().get(0);
              Hop incr = (fsb.getIncrementHops() != null) ?
                      fsb.getIncrementHops().getInput().get(0) : new LiteralOp(1);
  
              // Calculate for-loop iteration count (weight) if from, to, and incr are literal ops (constant values)
              if( from instanceof LiteralOp && to instanceof LiteralOp && incr instanceof LiteralOp ) {
                  double dfrom = HopRewriteUtils.getDoubleValue((LiteralOp) from);
                  double dto = HopRewriteUtils.getDoubleValue((LiteralOp) to);
                  double dincr = HopRewriteUtils.getDoubleValue((LiteralOp) incr);
                  if( dfrom > dto && dincr == 1 )
                      dincr = -1;
                  loopWeight = UtilFunctions.getSeqLength(dfrom, dto, dincr, false);
              }
              computeWeight *= loopWeight;
              networkWeight *= loopWeight;
 
             // 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
             List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
             currentLoopStack.add(Pair.of(sb.getSBID(), loopWeight));

            rewireHopDAG(fsb.getFromHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null,
                            innerTransTable, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack);
            rewireHopDAG(fsb.getToHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null,
                            innerTransTable, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack);

            if (fsb.getIncrementHops() != null) {
                rewireHopDAG(fsb.getIncrementHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null, innerTransTable, 
                                unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack);
            }
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, newFormerTransTable, 
                                            unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack));
        }
        else if (sb instanceof WhileStatementBlock) {
            WhileStatementBlock wsb = (WhileStatementBlock) sb;
            WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
           
            computeWeight *= DEFAULT_LOOP_WEIGHT;
            networkWeight *= DEFAULT_LOOP_WEIGHT;
            
            // 현재 루프 컨텍스트 생성 (부모 컨텍스트 복사)
            List<Pair<Long, Double>> currentLoopStack = new ArrayList<>(parentLoopStack);
            currentLoopStack.add(Pair.of(sb.getSBID(), DEFAULT_LOOP_WEIGHT));

            rewireHopDAG(wsb.getPredicateHops(), prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null, 
                            innerTransTable, unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerWsb : wstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerWsb, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, newFormerTransTable, 
                                            unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, currentLoopStack));
        }
        else if (sb instanceof FunctionStatementBlock) {
            FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
            FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, newFormerTransTable, 
                                            unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, parentLoopStack));
        }
        else { //generic (last-level)
            if( sb.getHops() != null ){
                for(Hop c : sb.getHops())
                    rewireHopDAG(c, prog, visitedHops, rewireTable, hopCommonTable, newOuterTransTableList, null, innerTransTable, 
                                    unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, parentLoopStack);
            }

            return innerTransTable;
        }
        return newFormerTransTable;
    }

    private static void rewireHopDAG(Hop hop, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable,  Map<Long, HopCommon> hopCommonTable, List<Map<String, List<Hop>>> outerTransTableList,
                                               Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<Long> unRefTwriteSet, Set<Long> unRefSet, Set<Hop> progRootHopSet, 
                                               Set<String> fnStack, double computeWeight, double networkWeight, List<Pair<Long, Double>> loopStack) {
       // Process all input nodes first if not already in memo table

        if (hop.getInput() != null){
            for (Hop inputHop : hop.getInput()) {
                long inputHopID = inputHop.getHopID();
                if (!visitedHops.contains(inputHopID)) {
                    visitedHops.add(inputHopID);
                    rewireHopDAG(inputHop, prog, visitedHops, rewireTable, hopCommonTable, outerTransTableList, formerTransTable, innerTransTable, 
                                    unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, loopStack);
                }
            }    
        }

        hopCommonTable.put(hop.getHopID(), new HopCommon(hop, computeWeight, networkWeight, 0, loopStack));

        // Identify hops to connect to the root dummy node
        // Connect TWrite pred and u(print) to the root dummy node
        if ((hop instanceof DataOp && (hop.getName().equals("__pred"))) // TWrite "__pred"
            || (hop instanceof UnaryOp && ((UnaryOp)hop).getOp() == Types.OpOp1.PRINT) // u(print)
            || (hop instanceof DataOp && ((DataOp)hop).getOp() == Types.OpOpData.PERSISTENTWRITE)){ // PWrite
            progRootHopSet.add(hop);
        } else if (!(hop instanceof DataOp && ((DataOp)hop).getOp() == Types.OpOpData.TRANSIENTWRITE)
                    && hop.getParent().size() == 0) {
            unRefSet.add(hop.getHopID());
        }

       if( hop instanceof FunctionOp )
       {
           //maintain counters and investigate functions if not seen so far
           FunctionOp fop = (FunctionOp) hop;
           if( fop.getFunctionType() == FunctionType.DML )
           {
               String fkey = fop.getFunctionKey();

               if(!fnStack.contains(fkey)) {
                   fnStack.add(fkey);
                   FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());

                   Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
                   if (formerTransTable != null){
                       newFormerTransTable.putAll(formerTransTable);
                   }
                   newFormerTransTable.putAll(innerTransTable);

                   String[] inputArgs = fop.getInputVariableNames();
                   List<Hop> inputHops = fop.getInput();

                   // functionTransTable에서 밖에 안 씀.
                   for (int i = 0; i < inputHops.size(); i++){
                       newFormerTransTable.computeIfAbsent(inputArgs[i], k -> new ArrayList<>()).add(inputHops.get(i));
                   }

                   // Todo (Future): 인자로 분리 안하면 RewireTable, MemoTable 분리해야 함.
                   Map<String, List<Hop>> functionTransTable = rewireStatementBlock(fsb, prog, visitedHops, rewireTable, hopCommonTable, outerTransTableList, newFormerTransTable, 
                                                                                        unRefTwriteSet, unRefSet, progRootHopSet, fnStack, computeWeight, networkWeight, loopStack);
                   
                   for (int i = 0; i < fop.getOutputVariableNames().length; i++){
                       String tWriteName = fop.getOutputVariableNames()[i];
                       List<Hop> outputHops = functionTransTable.get(fsb.getOutputsofSB().get(i).getName());
                       innerTransTable.computeIfAbsent(tWriteName, k -> new ArrayList<>()).addAll(outputHops);
                       for (Hop outputHop: outputHops){
                           unRefTwriteSet.add(outputHop.getHopID());
                       }
                   }
               }
           }
       }

       if (!(hop instanceof DataOp) || hop.getName().equals("__pred") || (hop instanceof DataOp && ((DataOp)hop).getOp() == Types.OpOpData.PERSISTENTWRITE)) {
           return;
       }

        rewireTransHop(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable, unRefTwriteSet);
   }

   private static void rewireTransHop(Hop hop, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList, Map<String, List<Hop>> formerTransTable,
                                    Map<String, List<Hop>> innerTransTable, Set<Long> unRefTwriteSet) {
       DataOp dataOp = (DataOp) hop;
       Types.OpOpData opType = dataOp.getOp();
       String hopName = dataOp.getName();

       if (opType == Types.OpOpData.TRANSIENTWRITE) {
           innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
           unRefTwriteSet.add(hop.getHopID());
       }
       else if (opType == Types.OpOpData.TRANSIENTREAD) {
            List<Hop> childHops = rewireTransRead(hopName, innerTransTable, formerTransTable, outerTransTableList);
            rewireTable.put(hop.getHopID(), childHops);
            
            if (childHops != null && !childHops.isEmpty()){
                for (Hop childHop: childHops){
                    rewireTable.computeIfAbsent(childHop.getHopID(), k -> new ArrayList<>()).add(hop);
                    unRefTwriteSet.remove(childHop.getHopID());
                }
            } else {
                System.out.println("hopName : " + hopName + " hop.getHopID() : " + hop.getHopID());
            }
       }
   }

   private static List<Hop> rewireTransRead(String hopName, Map<String, List<Hop>> innerTransTable,
                                                   Map<String, List<Hop>> formerTransTable, List<Map<String, List<Hop>>> outerTransTableList) {
       List<Hop> childHops = new ArrayList<>();

       // Read according to priority: inner -> former -> outer
       if (!innerTransTable.isEmpty()){
           childHops = innerTransTable.get(hopName);
       }

       if ((childHops == null || childHops.isEmpty()) && formerTransTable != null) {
           childHops = formerTransTable.get(hopName);
       }

       if (childHops == null || childHops.isEmpty()) {
           // 마지막으로 삽입된 outerTransTable부터 역순으로 순회
           for (int i = outerTransTableList.size() - 1; i >= 0; i--) {
               Map<String, List<Hop>> outerTransTable = outerTransTableList.get(i);
               childHops = outerTransTable.get(hopName);
               if (childHops != null && !childHops.isEmpty()) break;
           }
       }

       return childHops;
   }
} 