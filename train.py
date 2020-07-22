from TripleManager import TripleManager
from TransE import TransE
from TransH import TransH
from TransD import TransD
from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss
from Trainer import Trainer
import time
import sys

if __name__ == '__main__':
    #folder = sys.argv[1]
    #dataset = int((int(sys.argv[2])-1)/8)
    #negative_ent = 2**((int(sys.argv[2])-1)-(dataset*8))
    #corruptionMode = sys.argv[3]
    #modelName = sys.argv[4]
    #p_norm = int(sys.argv[5])
    #margin = float(sys.argv[6])

    folder = ""
    dataset = 2 # Options: 0-7
    negative_ent = 2 ** 0
    corruptionMode = "NLCWA" # Options: "Global" (naive), "LCWA", "TLCWA", "NLCWA", "GNLCWA"
    modelName = "transe" # Options: "transd", "transe", "transh"
    p_norm = 2
    margin = 5.0

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

    print("Dataset: " + dataset_name + "; NegRate: " + str(negative_ent) + \
          "; Mode: " + corruptionMode + "; Norm: " + str(p_norm))

    start = time.perf_counter()
    path = folder + "Datasets/" + dataset_name + "/"
    nbatches = 100
    negative_rel = 0
    train_manager = TripleManager(path, splits=["train"], nbatches=nbatches,
                                  neg_ent=negative_ent, neg_rel=negative_rel, corruptionMode=corruptionMode)

    transx = None
    if modelName=="transe":
        transx = TransE(
            ent_tot=train_manager.entityTotal,
            rel_tot=train_manager.relationTotal,
            dim=200,
            p_norm=p_norm,
            norm_flag=True)
    elif modelName == "transh":
        transx = TransH(
            ent_tot=train_manager.entityTotal,
            rel_tot=train_manager.relationTotal,
            dim=200,
            p_norm=p_norm,
            norm_flag=True)
    elif modelName == "transd":
        transx = TransD(
            ent_tot=train_manager.entityTotal,
            rel_tot=train_manager.relationTotal,
            dim_e=200,
            dim_r=200,
            p_norm=p_norm,
            norm_flag=True)

    model = NegativeSampling(
        model=transx,
        loss=MarginLoss(margin=margin),
        batch_size=train_manager.batchSize)
    end = time.perf_counter()
    print("Initialization time: " + str(end - start))

    start = time.perf_counter()
    trainer = Trainer(model=model, data_loader=train_manager, train_times=1500,
            alpha=1.0, use_gpu=False, save_steps=50,
            checkpoint_dir=folder + "Model/" + modelName + "_" + dataset_name + "_" + str(negative_ent) + "_" + \
                           corruptionMode + "_" + str(p_norm))
    trainer.run()
    end = time.perf_counter()
    print("Time elapsed during the calculation: " + str(end - start))