# %%
import webknossos as wk
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imread
import pandas as pd
import pickle
import numpy as np
from neurometry import ml
# %%
ds_path = "/home/tmn7/wkw-test/wkw2/binaryData/harvard-htem/"
ids = pd.read_csv("data/wk_ids_10_16_23.csv")

# %%
ids = ids[ids["purpose"] == "nuclei segmentation"]

# %%
id_to_annotation = {}
not_working = []
MAG = wk.Mag("1")
for index, row in ids.iterrows():
     #print(row["annotation_id"])
     ds = wk.dataset.Dataset('annotations/tmp', voxel_size=(100,100,100))
     img = wk.Dataset.open(ds_path + row["name"])
     bbox = img.get_layer("img").bounding_box
     try:
        with wk.webknossos_context(
                token="9ep0OkwPk41MnnKzqNGh9w",
                url="http://catmaid2.hms.harvard.edu:9000"):
            annotation = wk.Annotation.download(row["annotation_id"])
            annotation.export_volume_layer_to_dataset(ds)
        
        ds.get_segmentation_layers()[0].bounding_box = bbox
        mag_view = ds.get_segmentation_layers()[0].get_mag(MAG)
        data = mag_view.read()[0,:,:,:]
        id_to_annotation[row["name"]] = data
        ds.delete_layer('volume_layer')
     except:
        not_working.append(row["name"])

# %%

id_to_vol = ml.ingestor(ids)
# %%
shape_checker = []
for key, value in id_to_annotation.items():
    shape_checker.append([key, id_to_vol[key].shape, value.shape, 
                          id_to_vol[key].shape == value.shape])

for key, value in id_to_annotation.items():
    value[value != 0] = 1
    id_to_annotation[key] = np.flipud(np.rot90(value))
    
# %%
shape_checker_df = pd.DataFrame(shape_checker, columns=["wk_id", "vol_shape", "annotation_shape", "truth_value"])
# %%
with open('data/annotations_10_17_23.pickle', 'wb') as handle:
    pickle.dump(id_to_annotation, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
