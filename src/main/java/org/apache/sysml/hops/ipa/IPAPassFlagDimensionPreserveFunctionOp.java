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

package org.apache.sysml.hops.ipa;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.lops.FunctionCallCPSingle;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.StatementBlock;

/**
 * A rewrite to split DAGs after FunctionOps that return matrices/frames
 * and are not dimension-preserving.
 *
 */
public class IPAPassFlagDimensionPreserveFunctionOp extends IPAPass {
    @Override
    public boolean isApplicable() {
        return InterProceduralAnalysis.DIMENSION_PRESERVE_FUNCTIONOP;
    }

    @Override
    public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes )
            throws HopsException
    {
        if( !ConfigurationManager.isDynamicRecompilation() )
            return;

        try {
            for( String fkey : fgraph.getReachableFunctions() ) {
                FunctionOp first = fgraph.getFunctionCalls(fkey).get(0);

                //allow size propagation over FunctionOps during dynamic recompilation
                //TODO: find out how to propagate the size.
                FunctionStatementBlock fsblock = prog.getFunctionStatementBlocks(namespaceKey, fname);
                if( !fgraph.isRecursiveFunction(namespaceKey, fname) &&
                        rFlagDimensionPreserveFunctionOp(fsblock)) {
                    fsblock.setSplitDag(true);
                    if( LOG.isDebugEnabled() )
                        LOG.debug("IPA: FUNC flagged for hop dag-split:" +
                            DMLProgram.constructFunctionKey(namespaceKey, fname));
                }

            }
        }
        catch( LanguageException ex ) {
            throw new HopsException(ex);
        }
    }

    /**
     * Return true if this statement block requires recompilation.
     *
     * @param sb statement block
     * @return true if statement block requires recompilation
     */
    public boolean rFlagDimensionPreserveFunctionOp(StatementBlock sb ) {
        boolean ret = false;

        if (sb instanceof FunctionCallCPSingle) {
            //do something
        }
        else {
            ret |= ( sb.requiresRecompilation() );
        }

        return ret;
    }

}
