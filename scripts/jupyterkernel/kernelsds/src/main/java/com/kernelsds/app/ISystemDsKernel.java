package com.kernelsds.app;
import org.apache.sysds.api.jmlc.*;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.dml.DMLParserWrapper;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ParserFactory;

import io.github.spencerpark.jupyter.kernel.BaseKernel;
import io.github.spencerpark.jupyter.kernel.LanguageInfo;
import io.github.spencerpark.jupyter.kernel.ReplacementOptions;
import io.github.spencerpark.jupyter.kernel.display.DisplayData;
import io.github.spencerpark.jupyter.kernel.util.CharPredicate;
import io.github.spencerpark.jupyter.kernel.util.SimpleAutoCompleter;
import io.github.spencerpark.jupyter.kernel.util.StringSearch;

import javax.script.ScriptContext;
import javax.script.ScriptEngine;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class ISystemDsKernel extends BaseKernel{
    //private static final NashornScriptEngineFactory NASHORN_ENGINE_FACTORY = new NashornScriptEngineFactory();

    private static final SimpleAutoCompleter autoCompleter = SimpleAutoCompleter.builder()
            .preferLong()
            //Keywords from a great poem at https://stackoverflow.com/a/12114140
            .withKeywords("let", "this", "long", "package", "float")
            .withKeywords("goto", "private", "class", "if", "short")
            .withKeywords("while", "protected", "with", "debugger", "case")
            .withKeywords("continue", "volatile", "interface")
            .withKeywords("instanceof", "super", "synchronized", "throw")
            .withKeywords("extends", "final", "export", "throws")
            .withKeywords("try", "import", "double", "enum")
            .withKeywords("false", "boolean", "abstract", "function")
            .withKeywords("implements", "typeof", "transient", "break")
            .withKeywords("void", "static", "default", "do")
            .withKeywords("switch", "int", "native", "new")
            .withKeywords("else", "delete", "null", "public", "var")
            .withKeywords("in", "return", "for", "const", "true", "char")
            .withKeywords("finally", "catch", "byte")
            .build();

    private static final CharPredicate idChar = CharPredicate.builder()
            .inRange('a', 'z')
            .inRange('A', 'Z')
            .match('_')
            .build();


    private LanguageInfo languageInfo;
    private final Connection connection = new Connection();
    private int sca;
    private HashMap<String, Object> hashMap = new HashMap<>();

    public ISystemDsKernel() {

    }



    @Override
    public LanguageInfo getLanguageInfo() {
        return languageInfo;
    }


    /*String script =
                "  X = rand(rows=10, cols=10);"
                        + "R = matrix(0, rows=10, cols=1)"
                        + "parfor(i in 1:nrow(X))"
                        + "  R[i,] = sum(X[i,])"
                        + "print(sum(R))";*/


    void analyzeVariables(Hop hop, HashSet<String> inVars, HashSet<String> outVars) {
        // Check if the hop is a DataOp
        if (hop instanceof DataOp) {
            DataOp dataOp = (DataOp) hop;
            Types.OpOpData opType = dataOp.getOp();

            // If the hop is a TRANSIENTREAD (input variable), add to inVars
            if (opType == Types.OpOpData.TRANSIENTREAD) {
                inVars.add(hop.getName());
            }
            // If the hop is a TRANSIENTWRITE (output variable), add to outVars
            else if (opType == Types.OpOpData.TRANSIENTWRITE) {
                outVars.add(hop.getName());
            }
        }

        // Recursively analyze the input hops
        for (Hop inputHop : hop.getInput()) {
            analyzeVariables(inputHop, inVars, outVars);
        }
    }
    void analyzeStatementBlock(StatementBlock sb, HashSet<String> outVars) {
        // Extract variable names from 'updated' and 'kill' sets and add them to outVars
        if (sb.variablesUpdated() != null) {

            outVars.addAll(sb.variablesUpdated().getVariableNames());

        }

    }

    @Override
    public DisplayData eval(String expr) throws Exception {

        HashSet<String> inputVaraibles = new HashSet<>();
        HashSet<String> outputVaraibles = new HashSet<>();

        if (this.hashMap.size() > 0){
            int i = 0;
            for (String in: this.hashMap.keySet()
            ) {
                i++;
                expr =  in + "= read(\"./tmp/doesntexist" + i+ "\", data_type=\"scalar\", value_type=\"string\");\n"+ expr  ;
            }
            i = 0;

        }
       DMLParserWrapper dmlParser = new DMLParserWrapper();
        DMLProgram dml_program = dmlParser.parse(null,expr,null);
        DMLTranslator translator = new DMLTranslator(dml_program);
        translator.validateParseTree(dml_program);
        translator.liveVariableAnalysis(dml_program);
        translator.constructHops(dml_program);




        for (StatementBlock sb : dml_program.getStatementBlocks()) {
            //System.out.println(sb.toString());
            // Get HOPs for each statement block
            //ArrayList<Hop> hops = sb.getHops();
            //ArrayList<Lop> lops = sb.getLops();
            analyzeStatementBlock(sb, outputVaraibles);
        }


        String[] outputArray;
        //System.out.println("size: " + this.hashMap.size());
        //System.out.println("hash: " + this.hashMap.entrySet());
        if (this.hashMap.size() > 0){
            outputArray = outputVaraibles.toArray(new String[0]);

            for (String out: outputArray
            ) {
                //System.out.println("out: "+out);
                expr =  expr + ";\nwrite("+out+", './tmp/"+out+"');" ;
            }
            //System.out.println("expr: "+expr);
            //PreparedScript preparedScript = this.connection.prepareScript(expr,new String[]{"b"}, outputArray);
            PreparedScript preparedScript = this.connection.prepareScript(expr,this.hashMap.keySet().toArray(new String[0]), outputArray);
            //preparedScript.setScalar("b",7,true);
            //inScalar1 = read("./tmp/doesntexist1", data_type="scalar");
            //\nwrite(outString, './tmp/outString');
            for (String key: this.hashMap.keySet().toArray(new String[0])
                 ) {
                //System.out.println("key: "+key);
                preparedScript.setScalar(key, Integer.parseInt(this.hashMap.get(key).toString()));

            }

            ResultVariables res =  preparedScript.executeScript();
            for (String output: outputArray) {
                // Retrieve and store the output variable values
                Object value = res.getScalarObject(output);

                this.hashMap.put(output, value);
            }
        }
        else {
            outputArray = outputVaraibles.toArray(new String[0]);
            //new String[]{"Z"}
            for (String out: outputArray
                 ) {
                //System.out.println("out: "+out);
                expr =  expr + ";\nwrite("+out+", './tmp/"+out+"');"  ;
            }
            //System.out.println("expr: "+expr);
            PreparedScript preparedScript = this.connection.prepareScript(expr,new String[]{}, outputArray);


            //preparedScript.setScalar("X",10.0,true);
            //preparedScript.setScalar("X",7,true);
            ResultVariables res =  preparedScript.executeScript();
            for (String output: outputArray) {
                // Retrieve and store the output variable values

                Object value = res.getScalarObject(output);

                this.hashMap.put(output, value);
            }
        }



        //return new DisplayData(res.getString("Z"));
        //PreparedScript preparedScript = this.connection.prepareScript(expr,new String[]{}, new String[] {});
        //ResultVariables res =  preparedScript.executeScript();
        return null;
    }

    /*@Override
    public DisplayData inspect(String code, int at, boolean extraDetail) throws Exception {
        StringSearch.Range match = StringSearch.findLongestMatchingAt(code, at, idChar);
        String id = "";
        Object val = null;
        if (match != null) {
            id = match.extractSubString(code);
            val = this.engine.getContext().getAttribute(id);
        }

        return new DisplayData(val == null ? "No memory value for '" + id + "'" : val.toString());
    }*/

    @Override
    public ReplacementOptions complete(String code, int at) throws Exception {
        StringSearch.Range match = StringSearch.findLongestMatchingAt(code, at, idChar);
        if (match == null)
            return null;
        String prefix = match.extractSubString(code);
        return new ReplacementOptions(autoCompleter.autocomplete(prefix), match.getLow(), match.getHigh());
    }
}
