function sendMailExit(flag)
if flag==1
    command='echo "Process ended" | mail -a ./serverMatlabRunLog.txt -a ./serverMatlabRunLog.err -a ./results/pxADMM_fd_spmd_vars.png -a ./results/pxADMM_fd_spmd_steps.png -s "Program ended on server" p.zhai-1@student.tudelft.nl';
    system(command);
    exit;
elseif flag==2
    command='echo "Process ended" | mail -a ./serverMatlabRunLogPreEva.txt -a ./serverMatlabRunLogPreEva.err -a ./results/Agg/PerformanceEva/Graphs.png -a ./results/Agg/PerformanceEva/MeanRMSE.png -s "Program ended on server" p.zhai-1@student.tudelft.nl';
    system(command);
    exit;
end
end
