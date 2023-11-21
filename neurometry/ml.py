import pandas as pd
import numpy as np
from tifffile import imread
import os
import yaml
import skimage.measure
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


CONFIG_PATH = "input/"

class NucleiDataset(Dataset):
    def __init__(self, ids, volumes, annotations, transform=None, img_transform=None):
        self.samples = ids  # list the samples
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )

        self.volumes = volumes
        self.annotations = annotations
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.ToTensor()
                #already normalized
                #transforms.Normalize([0.5], [0.5]),
            ]
        )
    def get_sample_id(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)

    # fetch the training sample given its index
    def __getitem__(self, idx):
        #img_path = os.path.join(self.root_dir, self.samples[idx], "image.tif")
        #image = Image.open(img_path)
        image = self.volumes[self.samples[idx]]
        image = self.inp_transforms(image)
        mask = self.annotations[self.samples[idx]]
        mask = mask.astype(np.float32)
        mask = transforms.ToTensor()(mask)
        #mask_path = os.path.join(self.root_dir, self.samples[idx], "mask.tif")
        #mask = transforms.ToTensor()(Image.open(mask_path))
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)
        return image, mask
    

    
def show_random_dataset_image(dataset):
    idx = np.random.randint(0, len(dataset))  # take a random sample
    img, mask = dataset[idx]  # get the image and the nuclei masks
    f, axarr = plt.subplots(1, 3)  # make two plots on one figure
    axarr[0].imshow(img[0], cmap=plt.cm.gray)  # show the image
    axarr[1].imshow(mask[0], cmap=plt.cm.gray, interpolation=None)  # show the masks
    axarr[2].imshow(img[0]*mask[0], cmap=plt.cm.gray, interpolation=None)
    _ = [ax.axis("off") for ax in axarr]  # remove the axes
    print("Image size is %s" % {img[0].shape})
    print("ID is " + dataset.get_sample_id(idx))
    plt.show()

def get_train_loader(train_data, batch_size):
    return DataLoader(train_data, batch_size, shuffle=True)

def get_val_loader(val_data, batch_size):
    return DataLoader(val_data, batch_size)

def get_transform():
    return transforms.RandomCrop(512)

def ingestor(df):
    id_to_vol = {}
    for index, row in df.iterrows():
        name = row["name"]
        vol1 = np.transpose(imread("cutouts/"+name+"/img/vol.tiff"), (1,2,0))
        #seg1 = np.transpose(imread("cutouts/"+name+"/cellseg/cellseg.tiff"), (1,2,0))
        #masked1 = (seg1/255)*vol1
        #id_to_vol[name] = masked1
        #Divide by 255 here
        id_to_vol[name] = np.array(vol1/255.0).astype('float32')
    
    return id_to_vol

def collapsor(id_to_vol):
    collapsed_id_to_vol = {}
    for key, value in id_to_vol.items():
        z = value.shape[2]
        for i in range(z):
            _id = key + "_" + str(i)
            collapsed_id_to_vol[_id] = value[:,:,i]

    return collapsed_id_to_vol

def filter(id_to_annotation, object_size):
    id_to_annotation_cleaned = {}
    for key, value in id_to_annotation.items():
        id_to_annotation_cleaned[key] = np.array(value)
        for i in range(value.shape[2]):
            labeled_image, count = skimage.measure.label(value[:,:,i], return_num=True)
            objects = skimage.measure.regionprops(labeled_image)
            #major filtering step
            small_objects =[obj for obj in objects if obj.area<object_size]
            for j in small_objects:
                id_to_annotation_cleaned[key][j.bbox[0]:j.bbox[2], j.bbox[1]:j.bbox[3], i]=0
    return id_to_annotation_cleaned

#collapsed_id to cleaned annotation
def construct_annotations(collapsed_id_to_vol, annotations):
    collapsed_id_to_annotation = {}
    wkids = set()
    for key, value in collapsed_id_to_vol.items():
        substrings = key.split("_")
        original = substrings[0]
        index = substrings[1]
        collapsed_id_to_annotation[key] = annotations[original][:,:,int(index)]
        #wkids.add(original)
        wkids.add(key)
    #use the wkids set to filter out the wk_id_to_rating dataframe for stratification in model
    return collapsed_id_to_annotation, wkids

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
        return config

# %%
"""
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
"""
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def compute_pixel_metrics(collapsed_id_to_vol):
    pixel_sum = []
    for key, value in collapsed_id_to_vol.items():
        temp = []
        _strings = key.split("_")
        temp.append(key)
        temp.append(_strings[0])
        temp.append(_strings[1])
        temp.append(value.shape[0])
        temp.append(value.shape[1])
        temp.append(value.sum())
        temp.append(value.mean())
        temp.append(value.std())
        area =  (value > 0).sum() / (value.shape[0] * value.shape[1])
        temp.append(area)
        pixel_sum.append(temp)

    metrics_df = pd.DataFrame(pixel_sum, columns=["id", "original",
                                              "index", "length", "width",
                                              "sum", "mean", "std", "area_percentage"])
    
    incluster_metrics = metrics_df.groupby("original").agg({"mean" : "std", "std" : "std", "area_percentage" : "std"}).reset_index()
    incluster_metrics["sum_of_stds"] = incluster_metrics["mean"] + incluster_metrics["area_percentage"]
    return metrics_df, incluster_metrics

def cca(collapsed_id_to_vol, id_to_annotation):
    sections = []
    for key, value in collapsed_id_to_vol.items():
        temp = []
        _strings = key.split("_")
        temp.append(key)
        original = _strings[0]
        index = _strings[1]
        temp.append(original)
        temp.append(index)
        area =  (value > 0).sum()
        temp.append(area)
        labeled_image, count = skimage.measure.label(id_to_annotation[original][:,:,int(index)], return_num=True)
        temp.append(count)
        objects = skimage.measure.regionprops(labeled_image)
        object_areas = [obj["area"] for obj in objects]
        temp.append(object_areas)
        sections.append(temp)

    cca_df = pd.DataFrame(sections, columns=["id", "original",
                                                "index", "area", "nuclei_count",
                                                "nuclei_sizes"])
    # %%
    cca_df_grp = cca_df.groupby("original").agg({"area" : "sum", "nuclei_count" : ["sum", "std"]}).reset_index()
    # %%
    cca_df_grp.columns = cca_df_grp.columns.to_flat_index()
    lookup = {('original', ''): 'original', ('area', 'sum'): 'area_sum', 
            ('nuclei_count', 'sum') : 'nuclei_count_sum',
            ('nuclei_count', 'std') : 'nuclei_count_std'}
    cca_df_grp_result = cca_df_grp.rename(columns=lookup)
    cca_df_grp_result["nuclei_saturation"] = cca_df_grp_result["nuclei_count_sum"] / cca_df_grp_result["area_sum"]
    cca_df_grp_result["nuclei_saturation"] = cca_df_grp_result["nuclei_saturation"] * 1e6

    nuclei_sizes_df = cca_df[["id", "original", "index", "nuclei_sizes"]].explode("nuclei_sizes").reset_index()
    return cca_df, cca_df_grp, nuclei_sizes_df

class UNet(nn.Module):
    """UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1, depth=4, final_activation=None):
        super().__init__()

        assert depth < 10, "Max supported depth is 9"

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = depth

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(
                final_activation, nn.Module
            ), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList(
            [
                self._conv_block(in_channels, 16),
                self._conv_block(16, 32),
                self._conv_block(32, 64),
                self._conv_block(64, 128),
                self._conv_block(128, 256),
                self._conv_block(256, 512),
                self._conv_block(512, 1024),
                self._conv_block(1024, 2048),
                self._conv_block(2048, 4096),
            ][:depth]
        )
        # the base convolution block
        if depth >= 1:
            self.base = self._conv_block(2 ** (depth + 3), 2 ** (depth + 4))
        else:
            self.base = self._conv_block(1, 2 ** (depth + 4))
        # modules of the decoder path
        self.decoder = nn.ModuleList(
            [
                self._conv_block(8192, 4096),
                self._conv_block(4096, 2048),
                self._conv_block(2048, 1024),
                self._conv_block(1024, 512),
                self._conv_block(512, 256),
                self._conv_block(256, 128),
                self._conv_block(128, 64),
                self._conv_block(64, 32),
                self._conv_block(32, 16),
            ][-depth:]
        )

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(8192, 4096),
                self._upsampler(4096, 2048),
                self._upsampler(2048, 1024),
                self._upsampler(1024, 512),
                self._upsampler(512, 256),
                self._upsampler(256, 128),
                self._upsampler(128, 64),
                self._upsampler(64, 32),
                self._upsampler(32, 16),
            ][-depth:]
        )
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(16, out_channels, 1)
        self.activation = final_activation

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

#need to edit for more control later
def get_final_activation():
    return torch.nn.Sigmoid()

def get_loss_function():
    return nn.BCELoss()

def get_optimizer(model):
    return torch.optim.Adam(model.parameters())

def get_logger(name):
    return SummaryWriter(name)

def get_metric():
    return DiceCoefficient()

def save_weights(model, path):
    torch.save(model.state_dict(), path)

def load_weights(model, path):
    model.load_state_dict(torch.load(path))

    

# apply training for one epoch
def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break


# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)
    
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():

        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y).item()
            val_metric += metric(prediction>0.5, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )

class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / denominator.clamp(min=self.eps)