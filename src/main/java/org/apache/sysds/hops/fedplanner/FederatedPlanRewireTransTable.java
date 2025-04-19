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
    }

    public static void rewireFunctionDynamic(FunctionStatementBlock function, Map<Long, List<Hop>> rewireTable, Set<Long> unRefTwriteSet, Set<Hop> progRootHopSet) {
        Set<Long> visitedHops = new HashSet<>();
        Set<String> fnStack = new HashSet<>();

        List<Map<String, List<Hop>>> outerTransTableList = new ArrayList<>();
        Map<String, List<Hop>> outerTransTable = new HashMap<>();
        outerTransTableList.add(outerTransTable);
        // Todo: not tested
        rewireStatementBlock(function, null, visitedHops, rewireTable, outerTransTableList, null, unRefTwriteSet, progRootHopSet, fnStack);
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
                   FunctionStatementBlock newFsb = updateFunctionStatementBlockVariables(fop, fsb);
                   // Todo (Future): 인자로 분리 안하면 RewireTable, MemoTable 분리해야 함.
                   fop.setFunctionName(fkey);
                   prog.addFunctionStatementBlock(fkey, newFsb);

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

                   Map<String, List<Hop>> functionTransTable = rewireStatementBlock(newFsb, prog, visitedHops, rewireTable, outerTransTableList, newFormerTransTable, unRefTwriteSet, progRootHopSet, fnStack);
                   
                   for (int i = 0; i < fop.getOutputVariableNames().length; i++){
                       String tWriteName = fop.getOutputVariableNames()[i];
                       List<Hop> outputHops = functionTransTable.get(newFsb.getOutputsofSB().get(i).getName());
                       innerTransTable.computeIfAbsent(tWriteName, k -> new ArrayList<>()).addAll(outputHops);
                       for (Hop outputHop: outputHops){
                           unRefTwriteSet.add(outputHop.getHopID());
                       }
                   }
               }
           }
       }

        rewireTransReadWrite(hop, rewireTable, outerTransTableList, formerTransTable, innerTransTable, unRefTwriteSet);
   }

   private static void rewireTransReadWrite(Hop hop, Map<Long,List<Hop>> rewireTable, List<Map<String, List<Hop>>> outerTransTableList,
                                                   Map<String, List<Hop>> formerTransTable, Map<String, List<Hop>> innerTransTable, Set<Long> unRefTwriteSet) {
       if (!(hop instanceof DataOp) || hop.getName().equals("__pred") || (hop instanceof DataOp && ((DataOp)hop).getOp() == Types.OpOpData.PERSISTENTWRITE)) {
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
            rewireTable.put(hop.getHopID(), childHops);

            for (Hop childHop: childHops){
                rewireTable.computeIfAbsent(childHop.getHopID(), k -> new ArrayList<>()).add(hop);
                unRefTwriteSet.remove(childHop.getHopID());
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

   /**
    * FunctionOp의 입력 데이터 정보를 바탕으로 FunctionStatementBlock의 변수 정보를 업데이트합니다.
    * 
    * @param fop 함수 연산자
    * @param fsb 함수 구문 블록
    */
   private static FunctionStatementBlock updateFunctionStatementBlockVariables(FunctionOp fop, StatementBlock originalFsb) {
		// 새로운 FunctionStatementBlock 생성
       FunctionStatementBlock fsb = (FunctionStatementBlock) originalFsb.deepCopy();
       String[] inputArgs = fop.getInputVariableNames();
       List<Hop> inputHops = fop.getInput();
       
       for (int i = 0; i < inputHops.size(); i++) {
           Hop inputHop = inputHops.get(i);
           String argName = inputArgs[i];

           // 1. liveIn 변수 집합 업데이트
           if (fsb.liveIn().containsVariable(argName)) {
               DataIdentifier liveInVar = fsb.liveIn().getVariable(argName);
               liveInVar.setDimensions(inputHop.getDim1(), inputHop.getDim2());
               liveInVar.setNnz(inputHop.getNnz());
               liveInVar.setBlocksize(inputHop.getBlocksize());
               
               // 데이터 타입과 값 타입도 업데이트
               liveInVar.setDataType(inputHop.getDataType());
               liveInVar.setValueType(inputHop.getValueType());
           }

           // 2. liveOut 변수 집합 업데이트
           if (fsb.liveOut().containsVariable(argName)) {
               DataIdentifier liveOutVar = fsb.liveOut().getVariable(argName);
               liveOutVar.setDimensions(inputHop.getDim1(), inputHop.getDim2());
               liveOutVar.setNnz(inputHop.getNnz());
               liveOutVar.setBlocksize(inputHop.getBlocksize());
               liveOutVar.setDataType(inputHop.getDataType());
               liveOutVar.setValueType(inputHop.getValueType());
           }
           
           // 3. _gen 변수 집합 업데이트
           if (fsb.getGen() != null && fsb.getGen().containsVariable(argName)) {
               DataIdentifier genVar = fsb.getGen().getVariable(argName);
               genVar.setDimensions(inputHop.getDim1(), inputHop.getDim2());
               genVar.setNnz(inputHop.getNnz());
               genVar.setBlocksize(inputHop.getBlocksize());
               genVar.setDataType(inputHop.getDataType());
               genVar.setValueType(inputHop.getValueType());
           }
           
           // 4. _kill 변수 집합 업데이트
           if (fsb.getKill() != null && fsb.getKill().containsVariable(argName)) {
               DataIdentifier updatedVar = fsb.getKill().getVariable(argName);
               updatedVar.setDimensions(inputHop.getDim1(), inputHop.getDim2());
               updatedVar.setNnz(inputHop.getNnz());
               updatedVar.setBlocksize(inputHop.getBlocksize());
               updatedVar.setDataType(inputHop.getDataType());
               updatedVar.setValueType(inputHop.getValueType());
           }
       }

       DMLTranslator dmlt = new DMLTranslator(new DMLProgram());
       // Todo 더 복잡하게 해야할 듯...
       dmlt.constructHops(fsb);

       return fsb;
   }
} 