using DelimitedFiles

ps_telev1 = readdlm("Topologyv6/tele/data/ps_telev1.txt")
ps_telev2 = readdlm("Topologyv6/tele/data/datav2/ps_telev2.txt")

ps_tele_final = ps_telev1
ps_tele_final[:,11:20] = ps_telev2
ps_tele_final
writedlm("Topologyv6/tele/data/data_final/ps_tele_final.txt",ps_tele_final)

T1T2_list = [[0,20],[5,15],[10,10]]

for i = 1:length(T1T2_list)
    T1 = T1T2_list[i][1]
    T2 = T1T2_list[i][2]
    matrix_telev1 = readdlm("Topologyv6/tele/data/matrix_tele_$T1-$T2.csv")
    matrix_telev2 = readdlm("Topologyv6/tele/data/datav2/matrix_tele_$T1-$T2.csv")
    matrix_tele_final = matrix_telev1
    matrix_tele_final[:,11:20] = matrix_telev2
    writedlm("Topologyv6/tele/data/data_final/matrix_tele_final_$T1-$T2.txt",matrix_tele_final)
end





