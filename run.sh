#!/bin/bash
# start background matlab run
nohup matlab -nodisplay -nosplash -nodesktop -r "try;Main;sendMailExit;catch e;fprintf(1,'The identifier was:\n%s',e.identifier);fprintf(1,'There was an error! The message was:\n%s',e.message);sendErrMailExit;end" </dev/null >serverMatlabRunLog.txt 2>serverMatlabRunLog.err &
