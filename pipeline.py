# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from neurometry import ml
import seaborn as sns
from sklearn.model_selection import train_test_split
# %%
config = ml.load_config("config.yaml")
# %%
#ingestion part of the annotation from WK
with open(config["dataset"]["ann_path"], 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    id_to_annotation = pickle.load(f)

#metadata about the volume
df = pd.read_csv(config["dataset"]["vol_path"])
dff = df[df["purpose"] == "nuclei segmentation"]
id_to_vol = ml.ingestor(dff)
collapsed_id_to_vol = ml.collapsor(id_to_vol)

metrics_df, incluster_metrics = ml.compute_pixel_metrics(collapsed_id_to_vol)

# %%
cca_df, cca_df_grp, nuclei_sizes_df = ml.cca(collapsed_id_to_vol, id_to_annotation)

# %%
id_to_annotation_cleaned = ml.filter(id_to_annotation, 100)

# %%
cca_df_cleaned, cca_df_grp_cleaned, nuclei_sizes_df_cleaned = ml.cca(collapsed_id_to_vol, id_to_annotation_cleaned)

annotations, wkids = ml.construct_annotations(collapsed_id_to_vol, id_to_annotation_cleaned)

# %%
wkids = pd.DataFrame(wkids)

# %%
X_train, X_val = train_test_split(wkids, test_size=config["train"]["test_size"], random_state=config["train"]["random_state"])

# %%
TRAINING_DATA = list(X_train[0])
VALIDATION_DATA = list(X_val[0])
transform = ml.get_transform()
train_data = ml.NucleiDataset(TRAINING_DATA, collapsed_id_to_vol, annotations, transform)
val_data = ml.NucleiDataset(VALIDATION_DATA, collapsed_id_to_vol, annotations, transform)

# %%
train_loader = ml.get_train_loader(train_data, config["train"]["batch_size"])
val_loader = ml.get_val_loader(val_data, config["train"]["batch_size"])

final_activation = ml.get_final_activation()
loss_function = ml.get_loss_function()
logger = ml.get_logger(config["train"]["logger_path"]+config["train"]["id"])
model = ml.UNet(in_channels=1, out_channels=1, depth=config["train"]["depth"], final_activation=final_activation)
optimizer = ml.get_optimizer(model)
metric = ml.get_metric()

for epoch in range(config["train"]["epochs"]):
    ml.train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
    )

    step = epoch * len(train_loader.dataset)
    # validate
    ml.validate(
        model, 
        val_loader, 
        loss_function, 
        metric, 
        step=step, 
        tb_logger=logger)

ml.save_weights(model, config["train"]["weights_path"]+config["train"]["id"])

#plt.imshow(root_to_mask[720575940614448059][:,:,40], cmap=plt.cm.gray)
#plt.imshow(reshaped[:,:,0], cmap=plt.cm.gray)
#What would be included in the config file for the input?
#logging: yes or no
#optimizer: Adam
#metric: Dice Coefficient
#number of epochs
#loss function
#depth
#batch size
#random cropping
#config output to organize run data/parameters and names and connecrting back onto weights output

# %%

'''
# %%
sns.histplot(data=nuclei_sizes_df, x="nuclei_sizes")

# %%
sns.histplot(data=nuclei_sizes_df, )
# %%
sns.histplot(data=metrics_df, x="sum")
# %%
sns.histplot(data=metrics_df, x="mean")
# %%
sns.histplot(data=metrics_df, x="std")
# %%
sns.histplot(data=metrics_df, x="area_percentage")
# %%
sns.histplot(incluster_metrics, x="mean")
# %%
sns.histplot(incluster_metrics, x="area_percentage")
# %%
sns.histplot(incluster_metrics, x="std")
# %%
sns.histplot(incluster_metrics, x="sum_of_stds")
'''
