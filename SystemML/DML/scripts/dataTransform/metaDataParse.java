package com.ibm.systemML.dataTransformation;

import java.io.StringReader;

import com.ibm.jaql.json.type.JsonString;
import com.ibm.jaql.json.type.JsonValue;
import com.ibm.jaql.lang.core.Context;
import com.ibm.jaql.lang.expr.core.Expr;
import com.ibm.jaql.lang.parser.JaqlLexer;
import com.ibm.jaql.lang.parser.JaqlParser;

public class metaDataParse
{
        public JsonValue eval(JsonString str) throws Exception
        {
                JaqlLexer lexer = JaqlLexer.make(new StringReader(str.toString()));
                JaqlParser parser = new JaqlParser(lexer);
                Expr expr = parser.parse();

                return expr.eval(Context.newContext());
        }
}
