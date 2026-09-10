import sys
import subprocess
from antlr4 import FileStream, CommonTokenStream
from CASSLexer import CASSLexer
from CASSParser import CASSParser
from MyCASSVisitor import MyCassVisitor
from CASSNode import assign_usage_links

def drive_tree(input_file):
    #if len(sys.argv) < 2:
        #print("Usage: python driver.py <input_file>")
        #sys.exit(1)

    #input_file = sys.argv[1]

    # 1) Lex & parse
    input_stream = FileStream(input_file)
    lexer = CASSLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = CASSParser(token_stream)

    parse_tree = parser.prog()  # or whatever your top rule is

    # 2) Transform to CASS
    visitor = MyCassVisitor()
    cass_root = visitor.visit(parse_tree)
    assign_usage_links(cass_root)

    count = [] 
    for child in cass_root.children:
        count.append(child.get_node_count())
    cass_strings = cass_root.to_cass_string()  # Now returns a list
    final_list = []

    # **4) Print the results correctly**
   
    for i in range(len(cass_strings)):
        child = cass_root.children[i]

    # 1) Node count
        node_count = count[i]

        # 2) Source range
        src_range_str = child.get_source_range_string()

        # 3) The original CAS body
        cas_body = cass_strings[i]

        # 4) Combine them: "0,0,5,1    23    S#FS#1_2  ... rest ..."
        # Note the order: range -> node_count -> CAS body
        final_cas = f"{src_range_str}\t{node_count}\t{cas_body}"
        final_list.append(final_cas)
    
  
    return final_list
   

    # 4) Create DOT & PNG
    
    

if __name__ == "__main__":
    input_file = "input_code_ez.c"
    print(drive_tree(input_file))
