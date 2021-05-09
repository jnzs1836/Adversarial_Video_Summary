from data_loader import VideoImageData
import os
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from tqdm import tqdm
from feature_extraction import ResNetFeature
import sys
def main(root = "", save_path = "./out.h5"):
    hf = h5py.File(save_path, 'a')
    images_group = hf.create_group("pool5")
    model = ResNetFeature()
    for image_dir in os.listdir(root):
        image_group = images_group.create_group(image_dir)
        image_dir_path = os.path.join(root, image_dir)
        batch_size = 32
        loader = DataLoader(VideoImageData(image_dir_path), batch_size=batch_size)
        for i, batch in enumerate(tqdm(loader, desc="process: ")):
            batch_images = batch[0]
            batch_files = batch[1]
            batch_images = batch_images.cuda()
            features = model(batch_images)
            features = features[1].cpu()
            for j in range(batch_size):
                image_t = features[j, : ]
                image_t = torch.squeeze(image_t)
                image_data = image_t.detach().numpy()
                count = i * batch_size + j
                dset = image_group.create_dataset(batch_files[j],  data=image_data)


if __name__ == '__main__':
    main(sys.argv[1])
