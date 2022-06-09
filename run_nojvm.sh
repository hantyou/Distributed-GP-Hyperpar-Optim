#!/bin/bash
# start foreground matlab run
matlab -nosplash -nojvm -r "try;Main;sendMailExit;catch e;fprintf(1,'The identifier was:\n%s',e.identifier);fprintf(1,'There was an error! The message was:\n%s',e.message);sendErrMailExit;end"
