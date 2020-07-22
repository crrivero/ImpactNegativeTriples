from DataLoader import DataLoader
from TransE import TransE
from TransH import TransH
from TransD import TransD
import time
import sys
from Tester import Tester

if __name__ == '__main__':
    #folder = sys.argv[1]
    #dataset = int((int(sys.argv[2]) - 1) / 8)
    #negative_ent = 2 ** ((int(sys.argv[2]) - 1) - (dataset * 8))
    #corruptionModeModel = sys.argv[3]
    #corruptionModeTest = sys.argv[4]
    #modelName = sys.argv[5]
    #p_norm = int(sys.argv[6])
    #undesired = sys.argv[7]

    folder=""
    dataset=1
    negative_ent = 2**0
    corruptionModeModel="Global"
    corruptionModeTest="NLCWA"
    modelName = "transe"
    p_norm = 2
    undesired = "others"

    dataset_name = ""
    if dataset==0:
        dataset_name="FB13"
    if dataset==1:
        dataset_name="FB15K"
    if dataset==2:
        dataset_name="FB15K237"
    if dataset==3:
        dataset_name="NELL-995"
    if dataset==4:
        dataset_name="WN11"
    if dataset==5:
        dataset_name="WN18"
    if dataset==6:
        dataset_name="WN18RR"
    if dataset==7:
        dataset_name="YAGO3-10"

    print("Dataset: ", dataset_name, "; NegRate: ", str(negative_ent), "; Mode model: ", corruptionModeModel,
          "; Mode test:", corruptionModeTest, "; Model name:", modelName, "; Norm:", str(p_norm),"; Undesired:",undesired)

    path = folder + "Datasets/" + dataset_name + "/"
    nbatches = 100
    negative_rel = 0
    threads=8

    start = time.perf_counter()

    trainDataloader = DataLoader(path, "train")
    validDataloader = DataLoader(path, "valid")
    testDataloader = DataLoader(path, "test")

    undesiredRels = set()
    undesiredEnts = set()
    with open(path + "undesired_" + undesired + ".txt") as fp:
        line = fp.readline()
        if len(line.strip()) > 0:
            undesiredRels = undesiredRels.union(set([int(str) for str in line.strip().split(", ")]))
        line = fp.readline()
        if len(line.strip()) > 0:
            undesiredEnts = undesiredEnts.union(set([int(str) for str in line.strip().split(", ")]))

    print("DataLoaders are ready")

    norm_flag = False
    transx = None
    if modelName == "transe" or modelName.startswith("openke") or modelName.startswith("manual"):
        transx = TransE(
            ent_tot=trainDataloader.entityTotal,
            rel_tot=trainDataloader.relationTotal,
            dim=200,
            p_norm=p_norm,
            norm_flag=norm_flag)
    elif modelName == "transh":
        transx = TransH(
            ent_tot=trainDataloader.entityTotal,
            rel_tot=trainDataloader.relationTotal,
            dim=200,
            p_norm=p_norm,
            norm_flag=norm_flag)
    elif modelName == "transd":
        transx = TransD(
            ent_tot=trainDataloader.entityTotal,
            rel_tot=trainDataloader.relationTotal,
            dim_e=200,
            dim_r=200,
            p_norm=p_norm,
            norm_flag=norm_flag)

    modelFile = "Model/" + modelName + "_" + dataset_name + "_" + str(negative_ent) + "_" + corruptionModeModel + "_" + str(p_norm)
    transx.load_checkpoint(folder + modelFile + ".ckpt")

    print("Model is ready")

    tester = Tester(model=transx, train_data_loader=trainDataloader, valid_data_loader=validDataloader,
                    test_data_loader=testDataloader, undesired_ents=undesiredEnts, undesired_rels=undesiredRels,
                    use_gpu=False, corruptionMode=corruptionModeTest)
    tester.run()

    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))