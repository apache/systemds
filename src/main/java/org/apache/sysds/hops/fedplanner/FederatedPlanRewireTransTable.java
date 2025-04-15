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
    public static Map<Long, List<Hop>> rewireProgram(DMLProgram prog) {
        // Maps Hop ID and fedOutType pairs to their plan variants
        Map<Long, List<Hop>> rewireTable = new HashMap<>();

        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);
        Set<String> fnStack = new HashSet<>();
        Set<Long> visitedHops = new HashSet<>();

        for (StatementBlock sb : prog.getStatementBlocks()) {
           Map<String, List<Hop>> innerTransTable = rewireStatementBlock(sb, prog, visitedHops, rewireTable, outerTransTableList, null, fnStack);
           outerTransTableList.get(0).putAll(innerTransTable);
        }

        return rewireTable;
    }

    public static Map<String, List<Hop>> rewireStatementBlock(StatementBlock sb, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                                                Map<String, List<Hop>> formerTransTable, Set<String> fnStack) {
        List<Map<String, List<Hop>>> newOuterTransTableList = new ArrayList<>(outerTransTableList);

        if (formerTransTable != null){
            newOuterTransTableList.add(formerTransTable);
        }

        Map<String, List<Hop>> newFormerTransTable = new HashMap<>();
        Map<String, List<Hop>> innerTransTable = new HashMap<>();

        if (sb instanceof IfStatementBlock) {
            IfStatementBlock isb = (IfStatementBlock) sb;
            IfStatement istmt = (IfStatement)isb.getStatement(0);

            Map<String, List<Hop>> elseFormerTransTable = new HashMap<>();

            rewireHopDAG(isb.getPredicateHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);

            newFormerTransTable.putAll(innerTransTable);
            elseFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerIsb : istmt.getIfBody())
                newFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, fnStack));

            for (StatementBlock innerIsb : istmt.getElseBody())
                elseFormerTransTable.putAll(rewireStatementBlock(innerIsb, prog, visitedHops, rewireTable, newOuterTransTableList, elseFormerTransTable, fnStack));

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

            rewireHopDAG(fsb.getFromHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);
            rewireHopDAG(fsb.getToHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);
            rewireHopDAG(fsb.getIncrementHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, fnStack));
        }
        else if (sb instanceof WhileStatementBlock) {
            WhileStatementBlock wsb = (WhileStatementBlock) sb;
            WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);

            rewireHopDAG(wsb.getPredicateHops(), prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);
            newFormerTransTable.putAll(innerTransTable);

            for (StatementBlock innerWsb : wstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerWsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, fnStack));
        }
        else if (sb instanceof FunctionStatementBlock) {
            FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
            FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

            for (StatementBlock innerFsb : fstmt.getBody())
               newFormerTransTable.putAll(rewireStatementBlock(innerFsb, prog, visitedHops, rewireTable, newOuterTransTableList, newFormerTransTable, fnStack));
        }
        else { //generic (last-level)
            if( sb.getHops() != null ){
                for(Hop c : sb.getHops())
                    rewireHopDAG(c, prog, visitedHops, rewireTable, newOuterTransTableList, null, innerTransTable, fnStack);
            }

            return innerTransTable;
        }
        return newFormerTransTable;
    }

    private static void rewireHopDAG(Hop hop, DMLProgram prog, Set<Long> visitedHops, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                               Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<String> fnStack) {
       // Process all input nodes first if not already in memo table
       for (Hop inputHop : hop.getInput()) {
           long inputHopID = inputHop.getHopID();
           if (!visitedHops.contains(inputHopID)) {
               visitedHops.add(inputHopID);
               rewireHopDAG(inputHop, prog, visitedHops, rewireTable, outerTransTableList, formerTransTable, innerTransTable, fnStack);
           }
       }

       if( hop instanceof FunctionOp )
       {
           //maintain counters and investigate functions if not seen so far
           FunctionOp fop = (FunctionOp) hop;
           String fkey = fop.getFunctionKey();

           if( fop.getFunctionType() == FunctionType.DML )
           {
               FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
               // Todo: progRootHopSet, statRootHopSet을 이렇게 넘겨줘야하나?
               // Todo: 재귀랑 여러번 호출되는거랑 다른 것 아닌가?
               // Todo: Input/Output이 제대로 넘겨지는 것이 맞나?
                if(!fnStack.contains(fkey)) {
                    fnStack.add(fkey);
                    // Todo: function statement block은 내부적으로 또 if-else, loop 처리 해야함...
                    rewireStatementBlock(fsb, prog, visitedHops, rewireTable, outerTransTableList, null, fnStack);
                }
           }
       }

       // Determine modified child hops based on DataOp type and transient operations
        rewireTransReadWrite(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable);

   }

   private static void rewireTransReadWrite(Hop hop, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                                   Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable) {
       // TODO: How about PWrite?
       if (!(hop instanceof DataOp) || hop.getName().equals("__pred")) {
           return; // Early exit for non-DataOp or __pred
       }

       DataOp dataOp = (DataOp) hop;
       Types.OpOpData opType = dataOp.getOp();
       String hopName = dataOp.getName();

       if (opType == Types.OpOpData.TRANSIENTWRITE) {
           innerTransTable.computeIfAbsent(hopName, k -> new ArrayList<>()).add(hop);
       }
       else if (opType == Types.OpOpData.TRANSIENTREAD) {
           List<Hop> childHops = rewireTransRead(hopName,
               innerTransTable, formerTransTable, outerTransTableList);
           rewireTable.put(hop.getHopID(), childHops);

           for (Hop childHop: childHops){
               rewireTable.computeIfAbsent(childHop.getHopID(), k -> new ArrayList<>()).add(hop);
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
 