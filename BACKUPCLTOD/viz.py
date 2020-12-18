from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate
from collections import defaultdict
import collections


colors = sns.color_palette("Paired", 7)
colors_methods = {
                    "VANILLA":colors[6],
                    "L2":colors[2],
                    "EWC":colors[4],
                    "AGEM":colors[0],
                    "LAML":colors[5],
                    "REPLAY":colors[1],
                    "ADAPTER":colors[3],
                    "LOGADAPTER":colors[0]
                    }

def get_eval_from_metric(matrix_results,b,metric):
    percentage = False
    if "INTENT" == metric: 
        percentage = True
    if "DST-VALUE" == metric: 
        percentage = True
    if "DST-SLOT" == metric: 
        percentage = True
    if "EER" == metric: 
        percentage = True
    matrix_results = np.array(matrix_results)
    b = np.array(b)

    ACC = []
    for T in range(len(matrix_results)):
        temp = []
        for i in range(T+1):
            temp.append(matrix_results[T,i])
        if(percentage):
            ACC.append(round(np.mean(temp)*100,2))
        else:
            ACC.append(np.mean(temp))

    BWT = [0]
    for T in range(len(matrix_results)):
        temp = []
        for i in range(T):
            temp.append(matrix_results[T,i]- matrix_results[i,i])
        BWT.append(np.mean(temp))

    FWT = []
    for T in range(len(matrix_results)):
        temp = []
        for i in range(1,T+1):
            temp.append(matrix_results[i-1,i]- b[i])
        FWT.append(np.mean(temp))
    return ACC, BWT, FWT

def get_viz_folder(args,list_matrix_BLEU,list_b_INTENT,task_name,multi_task_row,methods_name,methods_name_multi,metric):
    list_results =[]
    table = [{"Method":methods_name_multi[i_r],"ACC":np.mean(row),"BWT":0,"FWT":0} for i_r, row in enumerate(multi_task_row)]
    for matrix, b, method in zip(list_matrix_BLEU,list_b_INTENT,methods_name):
        ACC, BWT, FWT = get_eval_from_metric(matrix, b, metric)
        table.append({"Method":method.replace("ADAPTER","ADAPTERCL"),"ACC":ACC[-1],"BWT":BWT[-1],"FWT":FWT[-1]})
        list_results.append([method,ACC, BWT, FWT])
    print(metric)
    print(tabulate(table, headers="keys"))
    x_axis_labels = [f"{t}" for t in range(len(task_name))] # labels for x-axis
    # x_axis_labels = task_name # labels for x-axis
    # y_axis_labels = task_name + [methods_name_multi[i_r] for i_r, row in enumerate(multi_task_row)] # labels for y-axis



    if(args.ablation):
        table_results_avg = defaultdict(list)
        for row in table:
            if("VANILLA" not in row["Method"] and "MULTI" not in row["Method"]):
                if(args.adapter):
                    size = int(row["Method"].split("BOTL_")[1].split("_")[0])
                else:
                    print(row["Method"].split("EM_")[1].split("_")[0])
                    size = int(row["Method"].split("EM_")[1].split("_")[0])
                    print(size)
                table_results_avg[size].append(row["ACC"])
        
        plt.figure(figsize=(6,4))
        y = []
        y_err = []
        x = []
        table_to_print = []
        for size, acc_y in collections.OrderedDict(sorted(table_results_avg.items())).items():
            if size == 100000: size = 1000
            if(args.adapter):
                x.append(int(size))
            else:
                x.append(int(size))

            y.append(float(np.mean(acc_y)))
            y_err.append(float(np.std(acc_y)))
            table_to_print.append({"Method":size,"ACC":f"{float(np.mean(acc_y))} +- {float(np.std(acc_y))}"})
        print(tabulate(table_to_print, headers="keys"))

        x = np.array(x)
        y = np.array(y)
        y_err = np.array(y_err)
        if(args.adapter):
            ax = plt.plot(x,y,label="ADAPTERCL")
        else:
            ax = plt.plot(x,y,label="REPLAY")
        plt.fill_between(x, y-y_err, y+y_err,alpha=0.3)#, edgecolor='#CC4F1B', facecolor='#FF9848')   
        for i_r, row in enumerate(multi_task_row):
            if "BLEU" != metric:
                plt.axhline(y=np.mean(row)*100, linestyle='--', lw=4,color="gray",alpha=0.6,label=methods_name_multi[i_r])
            else:
                plt.axhline(y=np.mean(row), linestyle='--', lw=4,color="gray",alpha=0.6,label=methods_name_multi[i_r])

        if(not args.adapter):
            xstr = list(x)  
            xstr[-1] = "ALL DATA"
            plt.xticks([i for i in x], [str(i) for i in xstr])

        if "INTENT" == metric: 
            metric = "INTENT-ACC"
            name = "INTENT"
        if "DST-VALUE" == metric: 
            metric = "JGA"
            name = "DST"
        if "DST-SLOT" == metric: 
            metric = "NAME-SLOT-ACC"
            name = "DST"
        if "BLEU" == metric: 
            name = "NLG"
        if "EER" == metric: 
            name = "EER"
        if(args.adapter):
            plt.xlabel('Adapter-Size')
            plt.ylabel(metric)
            plt.title(f"Adapter-Size vs Performance")
        else:
            plt.xlabel('Episodic Memory Size')
            plt.ylabel(metric)
            plt.title(f"Episodic Memory Size vs Performance")
        plt.legend()
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{args.model_checkpoint}/ABLATION_{metric}.png',dpi=300)
        plt.close()
    else:

        final_table_dic = defaultdict(list)
        final_table = []
        for row in table:
            if "MULTI" in row['Method']:
                final_table.append({"Method":"MULTI","ACC":round(row['ACC']*100,2)})
            else:
                final_table_dic[row['Method'].split("_")[0]].append(row['ACC'])

        for name, list_acc in final_table_dic.items():
            final_table.append({"Method":name,"ACC":f"${round(float(np.mean(list_acc)),2)} \pm {round(float(np.std(list_acc)),2)}$"})
        print(tabulate(final_table, headers="keys"))
        ### COMPUTE AVERAGE AND ERR AMONG PERMUTATION
        list_results_avg = defaultdict(list)
        for [method,ACC, BWT, FWT] in list_results:
            list_results_avg[method.split("_")[0]].append(ACC)


        plt.figure(figsize=(6,4))
            
        for method, ACC_mat in list_results_avg.items():
            mean = np.mean(ACC_mat, axis=0)
            std = np.std(ACC_mat, axis=0)
            plt.plot(mean,color=colors_methods[method],label="LAMOL" if method=="LAML" else method)
            plt.fill_between(range(len(mean)), mean-std, mean+std,alpha=0.1,facecolor=colors_methods[method])
            # if(method in ["REPLAY","ADAPTER"]):
            #     plt.fill_between(range(len(mean)), mean-std, mean+std,alpha=0.3, facecolor=colors_methods[method])
            # else:


        for i_r, row in enumerate(multi_task_row):
            if "BLEU" != metric:
                plt.axhline(y=np.mean(row)*100, linestyle='--', lw=4,color="gray",alpha=0.6,label=methods_name_multi[i_r])
            else:
                plt.axhline(y=np.mean(row), linestyle='--', lw=4,color="gray",alpha=0.6,label=methods_name_multi[i_r])
        plt.xticks([i for i in range(len(x_axis_labels))], [lab if i%2==0 else "" for i, lab in enumerate(x_axis_labels)])
        # plt.minorticks_on()
        plt.xlabel('Tasks Through Time')
        if "INTENT" == metric: 
            metric = "INTENT-ACC"
            name = "INTENT"
        if "DST-VALUE" == metric: 
            metric = "JGA"
            name = "DST"
        if "DST-SLOT" == metric: 
            metric = "NAME-SLOT-ACC"
            name = "DST"
        if "BLEU" == metric: 
            name = "NLG"
        if "EER" == metric: 
            name = "EER"
        plt.ylabel(metric)
        plt.title(f"{name}")
        plt.legend()
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{args.model_checkpoint}/ACC_results_{metric}.png',dpi=300)
        plt.close()



def get_viz(args,matrix_results,b,task_name,multi_task_row,metric):
    ACC, BWT, FWT = get_eval_from_metric(matrix_results,b)
    print(f"MULTI {metric}: {np.mean(multi_task_row)}")
    print(f"ACC {metric}: {ACC[-1]}")
    print(f"BWT {metric}: {BWT[-1]}")
    print(f"FWT {metric}: {FWT[-1]}")
    matrix_results = np.array(matrix_results)
    b = np.array(b)


    # create seabvorn heatmap with required labels
    x_axis_labels = task_name # labels for x-axis
    y_axis_labels = task_name + ["MULTI-TASK"] # labels for y-axis

    matrix_results = np.array(matrix_results.tolist() + [multi_task_row])
    plt.figure(figsize=(40,5))
    ax = sns.heatmap(matrix_results, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.xlabel('Tested Tasks')
    plt.ylabel(f"Trained Tasks'")
    plt.savefig(f'{args.model_checkpoint}/heatmap_results_{metric}.png')

    plt.close()

    plt.plot(ACC,label="T5")
    plt.axhline(y=np.mean(multi_task_row), linestyle='--',color="grey", lw=2,label="MULTI-TASK")
    plt.xticks([i for i in range(len(x_axis_labels))], x_axis_labels, rotation=45)
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.title("Metric vs Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{args.model_checkpoint}/ACC_results_{metric}.png')
    plt.close()

    plt.plot(BWT,label="T5")
    plt.xticks([i for i in range(len(x_axis_labels))], x_axis_labels, rotation=45)
    plt.xlabel('Time')
    plt.ylabel(f"BWT {metric}")
    plt.title("BWT Metric vs Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{args.model_checkpoint}/BWT_results_{metric}.png')
    plt.close()

    plt.plot(FWT,label="T5")
    plt.xticks([i for i in range(len(x_axis_labels))], x_axis_labels, rotation=45)
    plt.xlabel('Time')
    plt.ylabel(f"FWT {metric}")
    plt.title("FWT Metric vs Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{args.model_checkpoint}/FWT_results_{metric}.png')
    plt.close()
    return {"ACC":ACC,"BWT":BWT,"FWT":FWT}
