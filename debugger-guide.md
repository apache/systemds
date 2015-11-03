---
layout: global
title: SystemML Debugger Guide
description: SystemML Debugger Guide
---


## Overview

SystemML supports DML script-level debugging through a command line interface.  The SystemML debugger provides functionality typically found in a debugging environment like setting breakpoints, controlling program execution, and inspecting variables.  To run a script in debug mode, specify the '-debug' option as shown in below example.

    hadoop jar SystemML.jar -f test.dml -debug


## Debugger Commands

After starting a SystemML debug session, a list of available commands is automatically displayed.  Debugger commands can be entered at the SystemML debugger prompt (SystemMLdb).
The following sections describe each command along with example usage.

  * [Help](#help)
  * [Quit](#quit)
  * [List](#list)
  * [Step](#step)
  * [Break](#break)
  * [Delete](#delete)
  * [Info break](#info-break)
  * [Continue](#continue)
  * [Run](#run)
  * [Whatis](#whatis)
  * [Print](#print)
  * [Set](#set)
  * [Info frame](#info-frame)
  * [List instruction](#list-instruction)
  * [Step instruction](#step-instruction)



### Help

Type h for help to display a summary of available debugger commands.

    (SystemMLdb) h

    SystemMLdb commands:
    h,help                                                 list debugger functions
    r,run                                                  start your DML script
    q,quit                                                 exit debug mode
    c,continue                                             continue running your DML script
    l,list <[next numlines] | [prev numlines] | [all]>     display DML script source lines. Default: numlines = 10
    b,break <line-number>                                  set breakpoint at given line number
    d,delete <line-number>                                 delete breakpoint at given line number
    s,step                                                 next line, stepping into function calls
    i,info <break | frame>                                 show all breakpoints or frames (info <break | frame>)
    p,print <varName>                                      display contents of a scalar or matrix variable or
                                                           rows/columns/cell of matrix. (Eg: 'p alpha' or 'p A' or 'p A[1,]')
    set <varName value>                                    set value of a scalar or specified cell of a matrix variable. (Eg:
                                                           'set alpha 0.1' or 'set A[1,2] 20')
    whatis <varName>                                       display the type (and metadata) of a variable. (Eg: 'whatis alpha'
                                                           or 'whatis A')
    li,listi <[next numlines] | [prev numlines] | [all]>   display corresponding instructions for DML script source lines.
                                                           Default: numlines = 10  (for advanced users)
    si,stepi                                               next runtime instruction rather than DML source lines (for advanced
                                                           users)

    (SystemMLdb) 



### Quit

To exit a debug session, simply type q.

    q,quit                                                 exit debug mode

This returns control to the terminal or console shell which was used to launch the session.

    (SystemMLdb) q
    $


## Debugger commands for controlling script execution

This section describes commands that can be used to view and control script execution.  The following example script test.dml is used to demonstrate simple command usage scenarios.

    A = rand (rows=10, cols=5);
    B = rand (rows=5, cols=4);
    D = sum(A);
    print("Sum(A)=" + D);
    C = A %*% B;
    write(C, "output.csv", format="csv");




### List

After initially launching a debug session, the script is loaded and ready to be run.  The l command can be used to display the source lines of the loaded script.

    l,list <[next numlines] | [prev numlines] | [all]>     display DML script source lines. Default: numlines = 10

Without specifying any options, the list shows up to the next 10 lines of the script.  For example:

    (SystemMLdb) l
    line    1: A = rand (rows=10, cols=5);
    line    2: B = rand (rows=5, cols=4);
    line    3: D = sum(A);
    line    4: print("Sum(A)=" + D);
    line    5: C = A %*% B;
    line    6: write(C, "output.csv", format="csv");




### Step

Each line of the script can be stepped through using the s command.

    s,step                                                 next line, stepping into function calls

So continuing with the example from previous section, typing s executes the current line 1:

    (SystemMLdb) s
    Step reached at .defaultNS::main: (line 2).
    2    B = rand (rows=5, cols=4);
    (SystemMLdb) 

As can be seen from the output, the debugger executed line 1 and advanced to the next line in script.  The current line is automatically displayed.




### Break

To execute a group of instructions up to a specific line, breakpoints can be used. To set a breakpoint, use the b command.

    b,break <line-number>                                  set breakpoint at given line number


Continuing the example from step command, the current line was 2.  The below command sets a breakpoint at script source line number 4.

    (SystemMLdb) b 4    
    Breakpoint added at .defaultNS::main, line 4.
    (SystemMLdb) 




### Delete

Use the d command to remove a breakpoint.

    d,delete <line-number>                                 delete breakpoint at given line number

Below is sample output when removing a breakpoint.

    (SystemMLdb) d 4
    Breakpoint updated at .defaultNS::main, line 4.
    (SystemMLdb) 

If no breakpoint was set at the specified line number, then an appropriate message is displayed.

    (SystemMLdb) d 4
    Sorry, a breakpoint cannot be deleted at line 4. Please try a different line number.
    (SystemMLdb) 




### Info break

To see a list of breakpoints, use the i command with the break option.

    i,info break                                           show all breakpoints

Below is sample output after setting breakpoints at lines 2 and 4 of test.dml script.

    (SystemMLdb) i break
    Breakpoint  1, at line    2 (enabled)
    Breakpoint  2, at line    4 (enabled)
    (SystemMLdb) 

The info command also has a frame option which is discussed in the section related to inspecting script variables.

    i,info <break | frame>                                 show all breakpoints or frames (info <break | frame>)




### Continue

The continue command resumes script execution from the current line up to the next breakpoint.  If no breakpoints are set, then the rest of the script will be executed and the debugger session terminated.

    c,continue                                             continue running your DML script

Since the previous section set a breakpoint at line number 4, typing c to continue executes from the current line (2) up to but not including line 4 (i.e., the line with the breakpoint).

    (SystemMLdb) c
    Resuming DML script execution ...
    Breakpoint reached at .defaultNS::main instID 1: (line 4).
    4    print("Sum(A)=" + D);
    (SystemMLdb) 

Note that continue is not a valid command if the SystemML runtime has not been started.

    (SystemMLdb) c
    Runtime has not been started. Try "r" to start DML runtime execution.
    (SystemMLdb) 




### Run

There are two ways of starting the SystemML runtime for a debug session - the step command or the run command.  A common scenario is to set breakpoint(s) in the beginning of a debug session, then use r to start the runtime and run until the breakpoint is reached or script completion.

    r,run                                                  start your DML script

Using the same script from the previous example, the r command can be used in the beginning of the session to run the script up to a breakpoint or program completion if no breakpoint were set or reached.

    (SystemMLdb) l
    line    1: A = rand (rows=10, cols=5);
    line    2: B = rand (rows=5, cols=4);
    line    3: D = sum(A);
    line    4: print("Sum(A)=" + D);
    line    5: C = A %*% B;
    line    6: write(C, "output.csv", format="csv");
    (SystemMLdb) b 4
    Breakpoint added at .defaultNS::main, line 4.
    (SystemMLdb) r
    Breakpoint reached at .defaultNS::main instID 1: (line 4).
    4    print("Sum(A)=" + D);
    (SystemMLdb) 

Note the run command is not valid if the runtime has already been started.  In that case, use continue or step to execute line(s) of the script.

    (SystemMLdb) r
    Runtime has already started. Try "s" to go to next line, or "c" to continue running your DML script.
    (SystemMLdb) 


## Debugger Commands for inspecting or modifying script variables

Variables that are in scope can be displayed in multiple ways.  The same test.dml script is used for showing sample command usage.

    A = rand (rows=10, cols=5);
    B = rand (rows=5, cols=4);
    D = sum(A);
    print("Sum(A)=" + D);
    C = A %*% B;
    write(C, "output.csv", format="csv");




### Whatis

To display the type of a variable, use the whatis command.

    whatis <varName>                                       display the type (and metadata) of a variable. (Eg: 'whatis alpha'
                                                           or 'whatis A')

Given sample test.dml script with current line 4, then the metadata of variables A, B, D can be shown.

    (SystemMLdb) whatis A
    Metadata of A: matrix[rows = 10, cols = 5, rpb = 1000, cpb = 1000]
    (SystemMLdb) whatis B
    Metadata of B: matrix[rows = 5, cols = 4, rpb = 1000, cpb = 1000]
    (SystemMLdb) whatis D
    Metadata of D: DataType.SCALAR
    (SystemMLdb) 




### Print

To view the contents of a variable, use the p command.

    p,print <varName>                                      display contents of a scalar or matrix variable or
                                                           rows/columns/cell of matrix. (Eg: 'p alpha' or 'p A' or 'p A[1,]')

Below is sample print output for the same variables used in previous section.

    (SystemMLdb) p A
    0.6911	0.0533	0.7659	0.9130	0.1196	
    0.8153	0.6145	0.5440	0.2916	0.7330	
    0.0520	0.9484	0.2044	0.5571	0.6952	
    0.7422	0.4134	0.5388	0.1192	0.8733	
    0.6413	0.1825	0.4818	0.9019	0.7446	
    0.5984	0.8577	0.7151	0.3002	0.2228	
    0.0090	0.1429	0.2569	0.1421	0.1357	
    0.6778	0.8078	0.5075	0.0085	0.5159	
    0.8835	0.5621	0.7637	0.4362	0.4392	
    0.6108	0.5600	0.6140	0.0163	0.8640	
    (SystemMLdb) p B
    0.4141	0.9905	0.1642	0.7545	
    0.5733	0.1489	0.1204	0.5375	
    0.5202	0.9833	0.3421	0.7099	
    0.5846	0.7585	0.9751	0.1174	
    0.8431	0.5806	0.4122	0.3694	
    (SystemMLdb) p D
    D = 25.28558886582987
    (SystemMLdb) 

To display a specific element of a matrix, use [row,column] notation.

    (SystemMLdb) p A[1,1]
    0.6911
    (SystemMLdb) p A[10,5]
    0.8640
    (SystemMLdb)  

Specific rows or columns of a matrix can also be displayed.  The below examples show the first row and the fifth column of matrix A.

    (SystemMLdb) p A[1,]
    0.6911	0.0533	0.7659	0.9130	0.1196	
    (SystemMLdb) p A[,5]
    0.1196	
    0.7330	
    0.6952	
    0.8733	
    0.7446	
    0.2228	
    0.1357	
    0.5159	
    0.4392	
    0.8640	
    (SystemMLdb)




### Set

The set command is used for modifying variable contents.

    set <varName value>                                set value of a scalar or specified cell of a matrix variable. (Eg:
                                                       'set alpha 0.1' or 'set A[1,2] 20')

The following example modifies the first cell in matrix A.

    (SystemMLdb) set A[1,1] 0.3299
    A[1,1] = 0.3299
    (SystemMLdb)  

This example updates scalar D.  Note an equals sign is not needed when setting a variable.

    (SystemMLdb) set D 25.0
    D = 25.0
    (SystemMLdb) 




### Info frame

In addition to being used for displaying breakpoints, the i command is used for displaying frames.

    i,info frame                                       show all frames

So if our test.xml script was executed up to line 4, then the following frame information is shown.

    (SystemMLdb) i frame
    Current frame id: 0
      Current program counter at .defaultNS::main instID -1: (line 4)
      Local variables:
	    Variable name                            Variable value                          
	    A                                        Matrix: scratch_space//_p48857_9.30.252.162//_t0/temp1_1, [10 x 5, nnz=50, blocks (1000 x 1000)], binaryblock, dirty
	    B                                        Matrix: scratch_space//_p48857_9.30.252.162//_t0/temp2_2, [5 x 4, nnz=20, blocks (1000 x 1000)], binaryblock, dirty
	    D                                        25.28558886582987                       
    (SystemMLdb) 

Note only variables that are in scope are included (e.g., the variable C is not part of the frame since not yet in scope).


## Advanced Debugger Commands

This section describes commands for advanced users.  The same test.dml script is used for showing sample command usage.

    A = rand (rows=10, cols=5);
    B = rand (rows=5, cols=4);
    D = sum(A);
    print("Sum(A)=" + D);
    C = A %*% B;
    write(C, "output.csv", format="csv");




### List Instruction

The li command can be used to display lower-level instructions along with the source lines of the loaded script.

    li,listi <[next numlines] | [prev numlines] | [all]>   display corresponding instructions for DML script source lines.
                                                           Default: numlines = 10  (for advanced users)

For example:

    (SystemMLdb) li
    line    1: A = rand (rows=10, cols=5);
		 id   -1: CP createvar _mVar1 scratch_space//_p1939_9.30.252.162//_t0/temp1 true binaryblock 10 5 1000 1000 50
		 id   -1: CP rand 10 5 1000 1000 0.0 1.0 1.0 -1 uniform 1.0 4 _mVar1.MATRIX.DOUBLE
		 id   -1: CP cpvar _mVar1 A
		 id   -1: CP rmvar _mVar1
    line    2: B = rand (rows=5, cols=4);
		 id   -1: CP createvar _mVar2 scratch_space//_p1939_9.30.252.162//_t0/temp2 true binaryblock 5 4 1000 1000 20
		 id   -1: CP rand 5 4 1000 1000 0.0 1.0 1.0 -1 uniform 1.0 4 _mVar2.MATRIX.DOUBLE
		 id   -1: CP cpvar _mVar2 B
		 id   -1: CP rmvar _mVar2
    line    3: D = sum(A);
		 id   -1: CP uak+ A.MATRIX.DOUBLE _Var3.SCALAR.DOUBLE
		 id   -1: CP assignvar _Var3.SCALAR.DOUBLE.false D.SCALAR.DOUBLE
		 id   -1: CP rmvar _Var3
    line    4: print("Sum(A)=" + D);
		 id   -1: CP + Sum(A)=.SCALAR.STRING.true D.SCALAR.DOUBLE.false _Var4.SCALAR.STRING
		 id   -1: CP print _Var4.SCALAR.STRING.false _Var5.SCALAR.STRING
		 id   -1: CP rmvar _Var4
		 id   -1: CP rmvar _Var5
		 id   -1: CP rmvar D
    line    5: C = A %*% B;
		 id   -1: CP createvar _mVar6 scratch_space//_p1939_9.30.252.162//_t0/temp3 true binaryblock 10 4 1000 1000 -1
		 id   -1: CP ba+* A.MATRIX.DOUBLE B.MATRIX.DOUBLE _mVar6.MATRIX.DOUBLE 4
		 id   -1: CP cpvar _mVar6 C
		 id   -1: CP rmvar _mVar6
		 id   -1: CP rmvar A
		 id   -1: CP rmvar B
    line    6: write(C, "output.csv", format="csv");
		 id   -1: CP write C.MATRIX.DOUBLE output.csv.SCALAR.STRING.true csv.SCALAR.STRING.true false , false
		 id   -1: CP rmvar C
    (SystemMLdb) 




### Step Instruction

The si command can be used to step through the lower level instructions of an individual source line in a DML script.

    si,stepi                                               next runtime instruction rather than DML source lines (for advanced
                                                           users)

The first DML source line in test.dml consists of four instructions.


    (SystemMLdb) li next 0
    line    1: A = rand (rows=10, cols=5);
		 id   -1: CP createvar _mVar1 scratch_space//_p34473_9.30.252.162//_t0/temp1 true binaryblock 10 5 1000 1000 50
		 id   -1: CP rand 10 5 1000 1000 0.0 1.0 1.0 -1 uniform 1.0 4 _mVar1.MATRIX.DOUBLE
		 id   -1: CP cpvar _mVar1 A
		 id   -1: CP rmvar _mVar1
    (SystemMLdb) 

Type si to step through each individual instruction.

    (SystemMLdb) si
    Step instruction reached at .defaultNS::main instID -1: (line 1).
    1    A = rand (rows=10, cols=5);
    (SystemMLdb) si
    Step instruction reached at .defaultNS::main instID -1: (line 1).
    1    A = rand (rows=10, cols=5);
    (SystemMLdb) si
    Step instruction reached at .defaultNS::main instID -1: (line 1).
    1    A = rand (rows=10, cols=5);
    (SystemMLdb) si
    Step instruction reached at .defaultNS::main instID -1: (line 1).
    1    A = rand (rows=10, cols=5);
    (SystemMLdb)

Typing si again starts executing instructions of the next DML source line.

    (SystemMLdb) si
    Step instruction reached at .defaultNS::main instID -1: (line 2).
    2    B = rand (rows=5, cols=4);
    (SystemMLdb)

* * *
