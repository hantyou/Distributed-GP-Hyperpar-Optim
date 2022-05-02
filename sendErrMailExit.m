command='echo "Process ended with err" | mail -a ./serverMatlabRunLog.txt -a ./serverMatlabRunLog.err -s "Program err on server" p.zhai-1@student.tudelft.nl';
system(command);
exit;
