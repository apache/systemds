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
import org.apache.sysds.runtime.instructions.cp.ScalarObject;

import javax.script.ScriptContext;
import javax.script.ScriptEngine;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
    private HashMap<String, double[][]> hashMapForMatrix = new HashMap<>();

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

    public double[][] stringToDoubleArray(String matrixString) {
        // Split the input string into rows.
        String[] rows = matrixString.split(";");

        // Prepare the resulting double array with the correct number of rows.
        double[][] matrix = new double[rows.length][];

        for (int i = 0; i < rows.length; i++) {
            // Split each row into its individual values.
            String[] values = rows[i].split(",");

            // Prepare the row array.
            matrix[i] = new double[values.length];

            for (int j = 0; j < values.length; j++) {
                // Convert each value to double and store in the matrix.
                try {
                    matrix[i][j] = Double.parseDouble(values[j].trim()); // Trim to remove any leading or trailing spaces.
                } catch (NumberFormatException e) {
                    // Handle the case where the string cannot be parsed as a double.
                    System.err.println("Error parsing '" + values[j] + "' as double.");
                    // Optionally, initialize to a default value or throw an exception.
                }
            }
        }

        return matrix;
    }


    public static Set<String> extractOutputVariables(String script) {
        Set<String> outputVariables = new HashSet<>(); // Use a set to avoid duplicates
        String[] lines = script.split("\\n"); // Split the script into lines

        // Pattern to extract variable names (considering words before '=' and ignoring whitespace)
        Pattern pattern = Pattern.compile("\\s*(\\w+)\\s*(?:\\[.*\\])?\\s*=.*");

        for (String line : lines) {
            Matcher matcher = pattern.matcher(line);
            if (matcher.find()) {
                outputVariables.add(matcher.group(1)); // Add the variable name to the set
            }
        }

        return outputVariables;
    }

    public static Set<String> findInputVariables(String script) {
        // Split the script into lines and then into individual statements
        String[] statements = script.replace("\n", ";").split(";");

        Set<String> definedVariables = new HashSet<>();
        Set<String> usedVariables = new HashSet<>();

        // Pattern to match identifiers that are not purely numeric and not followed by '(' (to exclude function calls)
        // and not enclosed in quotes.
        Pattern variablePattern = Pattern.compile("\\b[a-zA-Z]\\w*\\b(?!\\()");
        Pattern stringPattern = Pattern.compile("\"[^\"]*\"|'[^']*'");

        for (String statement : statements) {
            statement = statement.trim(); // Trim whitespace from each statement

            // Remove string literals from the statement before analyzing
            Matcher stringMatcher = stringPattern.matcher(statement);
            String statementWithoutStrings = stringMatcher.replaceAll("");

            // Check for variable definitions and uses in the modified statement
            Matcher varMatcher = variablePattern.matcher(statementWithoutStrings);
            while (varMatcher.find()) {
                usedVariables.add(varMatcher.group());
            }

            // Check for variable definitions (assuming simple assignment statements)
            String[] parts = statementWithoutStrings.split("=");
            if (parts.length == 2) {
                String potentialVariable = parts[0].trim();
                if (variablePattern.matcher(potentialVariable).matches()) {
                    definedVariables.add(potentialVariable);
                }
            }
        }

        // Input variables are those that are used but not defined within the script
        usedVariables.removeAll(definedVariables);

        return usedVariables;
    }

    public static Object createObject(String type, ScalarObject value) {
        System.out.println("type: "+type);
        switch (type) {
            case "class org.apache.sysds.runtime.instructions.cp.StringObject":
                return value.getStringValue();
            case "class org.apache.sysds.runtime.instructions.cp.BooleanObject":
                return value.getBooleanValue();
            case "class org.apache.sysds.runtime.instructions.cp.DoubleObject":
                return value.getDoubleValue();
            case "class org.apache.sysds.runtime.instructions.cp.IntObject":
                return Integer.parseInt(value.getStringValue());
            default:
                throw new IllegalArgumentException("Unsupported type: " + type);
        }
    }

    /*void analyzeVariables(Hop hop, HashSet<String> inVars, HashSet<String> outVars) {
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
    }*/
    void analyzeStatementBlock(StatementBlock sb, HashSet<String> outVars) {
        // Extract variable names from 'updated' and 'kill' sets and add them to outVars
        if (sb.variablesUpdated() != null) {
            outVars.addAll(sb.variablesUpdated().getVariableNames());
        }
    }

    @Override
    public DisplayData eval(String expr) throws Exception {

        //HashSet<String> inputVariables = new HashSet<>();
        //HashSet<String> outputVariables = new HashSet<>();
        String[] outputArray;
        Set<String> set1 = findInputVariables(expr);
        Set<String> setoutputs = findInputVariables(expr);




       /*DMLParserWrapper dmlParser = new DMLParserWrapper();
        DMLProgram dml_program = dmlParser.parse(null,expr,null);
        DMLTranslator translator = new DMLTranslator(dml_program);
        translator.validateParseTree(dml_program);
        translator.liveVariableAnalysis(dml_program);
        translator.constructHops(dml_program);




        for (StatementBlock sb : dml_program.getStatementBlocks()) {
            analyzeStatementBlock(sb, outputVariables);
        }*/
        int i = 0;
        for (String str: set1
        ) {
            i++;
            if(this.hashMapForMatrix.get(str) != null){
                expr =  str + "= read(\"./tmp/doesntexist" + str+ "\", data_type=\"matrix\");\n"+ expr  ;
            }
            else{
                expr =  str + "= read(\"./tmp/doesntexist" + i+ "\", data_type=\"scalar\", value_type=\"string\");\n"+ expr  ;
            }
        }
        //i = 0;

        if (this.hashMap.size() > 0 || this.hashMapForMatrix.size()>0){
            setoutputs.addAll(extractOutputVariables(expr));
            setoutputs.addAll(set1);
            outputArray = setoutputs.toArray(new String[0]);

            for (String out: outputArray
            ) {
                //System.out.println("out: "+out);
                expr =  expr + ";\nwrite("+out+", './tmp/"+out+"');" ;
            }
            System.out.println("expr: "+expr);
            //PreparedScript preparedScript = this.connection.prepareScript(expr,new String[]{"b"}, outputArray);
            PreparedScript preparedScript = this.connection.prepareScript(expr,set1.toArray(new String[0]), outputArray);
            //preparedScript.setScalar("b",7,true);
            //inScalar1 = read("./tmp/doesntexist1", data_type="scalar");
            //\nwrite(outString, './tmp/outString');
            for (String key: set1
                 ) {
                System.out.println("hello: " + key);
                if(this.hashMapForMatrix.get(key) != null){

                    double [][] matrix = this.hashMapForMatrix.get(key);

                    preparedScript.setMatrix(key, matrix);

                }
                else {
                    switch (this.hashMap.get(key).getClass().getSimpleName()) {
                        case "String":
                            preparedScript.setScalar(key, this.hashMap.get(key).toString());
                            break; // Prevent fall-through
                        case "Boolean":
                            preparedScript.setScalar(key, Boolean.parseBoolean(this.hashMap.get(key).toString()));
                            break; // Prevent fall-through
                        case "Double":
                            preparedScript.setScalar(key, Double.parseDouble(this.hashMap.get(key).toString()));
                            break; // Prevent fall-through
                        case "Integer":
                            preparedScript.setScalar(key, Integer.parseInt(this.hashMap.get(key).toString()));
                            break; // Prevent fall-through

                        default:
                            throw new IllegalArgumentException("Unsupported type " + this.hashMap.get(key).getClass().getSimpleName());
                    }
                }


            }


            ResultVariables res =  preparedScript.executeScript();
            //System.out.println(res.);
            for (String output: outputArray) {
                // Retrieve and store the output variable values
                Types.DataType dt = res.getDataType(output);
                System.out.println("output is: " + output);
                System.out.println("datatype is: " + dt);
                if(dt.isMatrix()){
                    this.hashMapForMatrix.put(output, res.getMatrix(output));
                }

                else{
                    ScalarObject tmp = res.getScalarObject(output);

                    Object value = createObject(tmp.getClass().toString(), tmp);
                    this.hashMap.put(output, value);
                }

            }
        }
        else {
            setoutputs.addAll(extractOutputVariables(expr));
            setoutputs.addAll(set1);
            outputArray = setoutputs.toArray(new String[0]);
            //new String[]{"Z"}
            for (String out: outputArray
                 ) {
                System.out.println("out: "+out);
                expr =  expr + ";\nwrite("+out+", './tmp/"+out+"');"  ;
            }
            System.out.println("expr: "+expr);
            PreparedScript preparedScript = this.connection.prepareScript(expr,new String[]{}, outputArray);


            //preparedScript.setScalar("X",10.0,true);
            //preparedScript.setScalar("X",7,true);
            ResultVariables res =  preparedScript.executeScript();
            for (String output: outputArray) {
                // Retrieve and store the output variable values
                Types.DataType dt = res.getDataType(output);
                System.out.println("output is: " + output);
                System.out.println("datatype is: " + dt);
                if(dt.isMatrix()){

                    //Object value = res.getMatrix(output);
                    //System.out.println(res.getMatrix(output).length);

                    this.hashMapForMatrix.put(output, res.getMatrix(output));


                }
                else{
                    ScalarObject tmp = res.getScalarObject(output);
                    Object value = createObject(tmp.getClass().toString(), tmp);
                    System.out.println("value: "+value);
                    this.hashMap.put(output, value);
                }



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
