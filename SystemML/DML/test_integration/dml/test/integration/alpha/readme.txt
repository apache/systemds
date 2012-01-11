Java files which are placed in this package (dml.test.integration.alpha) 
will not be compiled (or tested) by ant when using ant auto or ant build.

Usecase:
You add new libraries to the project (and therewith also to the buildpath) to play around with their functionality. 
Then you may choose this package, sothat you can use ant for the rest of the project correctly (otherwise you would
have to add the library to ant's buildpath too)