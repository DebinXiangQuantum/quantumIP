numvars = 4
import os
import json
with open("bosonicqaoa.csv",'w') as f:
    f.write("num_vars,num_cons,problemid,p,ARG,success_rate,incons_rate\n")
    for numcons in [1,2]:
        for problemid in [0,1]:
            problem_dir = f"vars{numvars}/cons{numcons}/problem_instance_{problemid}"
            for p in range(1,7):
                path = os.path.join(problem_dir,f"khosravi_vqa_results_p{p}.json")
                if os.path.exists(path):

                    
                    with open(path, 'r', encoding='utf-8') as dataf:
                        data = json.load(dataf)
                        f.write(",".join(map(lambda x:str(x),[numvars,numcons,problemid,p,data['ARG'],data['success_rate'],
                            data['feasible_probability_sum']]))+"\n")