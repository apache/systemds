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

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.*;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.parser.*;
import java.util.*;

public class FederatedPlanRewireTransTable {
    public static void rewireProgram(DMLProgram prog, Map<Long, List<Hop>> rewireTable, Set<Long> unRefTwriteSet, Set<Hop> progRootHopSet) {
        // Maps Hop ID and fedOutType pairs to their plan variants
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();

        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);

        for (StatementBlock sb : prog.getStatementBlocks()) {
           Map<String, List<Hop>> innerTransTable = rewireStatementBlock(sb, prog, visitedHops, rewireTable, outerTransTableList, null, unRefTwriteSet, progRootHopSet, fnStack);
           outerTransTableList.get(0).putAll(innerTransTable);
        }

        return;
    }

    public static Map<String, List<Hop>> rewireStatementBlock(StatementBlock sb, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                                                Map<String, List<Hop>> formerTransTable, Set<Long> unRefTwriteSet, Set<Hop> progRootHopSet, Set<String> fnStack) {
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

            Map<String, List<Hop>> elseFormerTransTable = new HashMap<>();

            rewireHopDAG(isb.getPredicateHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);

            newFormerTransTable.putAll(innerTransTable);
            elseFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerIsb : istmt.getIfBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack));

            for (StatementBlock innerIsb : istmt.getElseBody())
                elseFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, newOuterTransTableList, elseFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack));

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

            rewireHopDAG(fsb.getFromHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
            rewireHopDAG(fsb.getToHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
            rewireHopDAG(fsb.getIncrementHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack));
        }
        else if (sb instanceof WhileStatementBlock) {
            WhileStatementBlock wsb = (WhileStatementBlock) sb;
            WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);

            rewireHopDAG(wsb.getPredicateHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerWsb : wstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerWsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack));
        }
        else if (sb instanceof FunctionStatementBlock) {
            FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
            FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack));
        }
        else { //generic (last-level)
            if( sb.getHops() != null ){
                for(Hop c : sb.getHops())
                    rewireHopDAG(c, prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
            }

            return innerTransTable;
        }
        return newFormerTransTable;
    }

    private static void rewireHopDAG(Hop hop, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                               Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<Long> unRefTwriteSet, Set<Hop> progRootHopSet, Set<String> fnStack) {
       // Process all input nodes first if not already in memo table
       for (Hop inputHop : hop.getInput()) {
           long inputHopID = inputHop.getHopID();
           if (!visitedHops.contains(inputHopID)) {
               visitedHops.add(inputHopID);
               rewireHopDAG(inputHop, prog, visitedHops, rewireTable, outerTransTableList, formerTransTable, innerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
           }
       }
    
        // Identify hops to connect to the root dummy node
        // Connect TWrite pred and u(print) to the root dummy node
        if ((hop instanceof DataOp && (hop.getName().equals("__pred"))) // TWrite "__pred"
            || (hop instanceof UnaryOp && ((UnaryOp)hop).getOp() == Types.OpOp1.PRINT) // u(print)
            || (hop instanceof DataOp && ((DataOp)hop).getOp() == Types.OpOpData.PERSISTENTWRITE)){ // PWrite
            progRootHopSet.add(hop);
        }

       if( hop instanceof FunctionOp )
       {
           //maintain counters and investigate functions if not seen so far
           FunctionOp fop = (FunctionOp) hop;
           if( fop.getFunctionType() == FunctionType.DML )
           {
               String fkey = fop.getFunctionKey();
               for (Hop inputHop : fop.getInput()){
                   fkey += "," + inputHop.getName();
               }

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

                   // Todo: Input에 따른 Cost(Memory Estimation) 반영 안됨 -> 다른 Input 동일 Cost
                   Map<String, List<Hop>> functionTransTable = rewireStatementBlock(fsb, prog, visitedHops, rewireTable, outerTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
                   String tWriteName = fop.getOutputVariableNames()[0];
                   List<Hop> outputHops = functionTransTable.get(fsb.getOutputsofSB().get(0).getName());
                   innerTransTable.computeIfAbsent(fop.getOutputVariableNames()[0], k -> new ArrayList<>()).addAll(outputHops);
                   // Todo: 이건 어떻게 등록하지?
                   // unRefTwriteSet.add(fop.getOutputVariableNames()[0]);
               }
           }
       }

       // Determine modified child hops based on DataOp type and transient operations
        rewireTransReadWrite(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable, unRefTwriteSet);
   }

   private static void rewireTransReadWrite(Hop hop, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                                   Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<Long> unRefTwriteSet) {
       // TODO: How about PWrite?
       if (!(hop instanceof DataOp) || hop.getName().equals("__pred")) {
           return; // Early exit for non-DataOp or __pred
       }

       DataOp dataOp = (DataOp) hop;
       Types.OpOpData opType = dataOp.getOp();
       String hopName = dataOp.getName();

       if (opType == Types.OpOpData.TRANSIENTWRITE) {
           innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
           unRefTwriteSet.add(hop.getHopID());
       }
       else if (opType == Types.OpOpData.TRANSIENTREAD) {
           List<Hop> childHops = rewireTransRead(hopName, innerTransTable, formerTransTable, outerTransTableList);
           // Todo 정상적인 상황이 아님 (재귀함수인 경우는 어쩔 수 없음. 나머지는...? 함수인 경우에만 표시해서 패스?)
           if (childHops != null){
               rewireTable.put(hop.getHopID(), childHops);

               for (Hop childHop: childHops){
                   rewireTable.computeIfAbsent(childHop.getHopID(), k -> new ArrayList<>()).add(hop);
                   unRefTwriteSet.remove(childHop.getHopID());
               }
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
 