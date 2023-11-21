# %%
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from neurometry import ml
# %%
raw = np.transpose(imread("cutouts/monkeyv1nucleiseg003/img/vol.tiff"), (1,2,0))
# %%
#raw = np.transpose(raw)
plt.imshow(raw[:,:,0], cmap=plt.cm.gray)
# %%
config = ml.load_config("config.yaml")
#raw = np.flipud(np.rot90(raw))
# %%
final_activation = ml.get_final_activation()
model = ml.UNet(in_channels=1, out_channels=1, depth=config["train"]["depth"], final_activation=final_activation)

# %%
keys = []
for key, value in model.state_dict().items():
    keys.append(key)
    print(key, '\n', value)
#import pickle
#with open("data/annotations_10_17_23.pickle", 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
#    id_to_annotation = pickle.load(f)
# %%
ml.load_weights(model, config["train"]["weights_path"]+config["train"]["id"])
# %%
for key, value in model.state_dict().items():
    print(key, '\n', value)
# %%
model.eval()
# %%
import copy
tester = copy.deepcopy(raw[:,:,0])
# %%
tester = np.array(tester/255.0).astype('float32')
# %%
from torchvision import transforms
# %%
tester_tensor = transforms.ToTensor()(tester)
# %%
import torch
import torch.nn as nn
# %%
pad = nn.ZeroPad2d(12)
# %%
output = pad(tester_tensor)
# %%
#model(output[None, :,:,:])[0,0,:,:].detach().numpy()
#plt.imshow(model(output[None, :,:,:])[0,0,:,:].detach().numpy(), cmap=plt.cm.gray)
#1024 slices are working quite well
#depad = nn.ZeroPad2d(-12)
#plt.imshow(depad(model(output[None, :,:,:]))[0,0,:,:].detach().numpy(), cmap=plt.cm.gray)
#basically ghave to have the extra channel and a multiple of 2 to do the convolutions 