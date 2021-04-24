from torch.utils.data import Dataset

class IMGDataset(Dataset):
    def __init__(self, config, tfm):

        self.img_list = os.listdir(args.photos_path)
        self.monet_list = os.listdir(args.paintings_path)
        self.tfm = tfm
        
    def __len__(self):
        return min(len(self.img_list), len(self.monet_list))
    
    def __getitem__(self, idx):

        img_index = idx
        monet_index = idx
        
        img = Image.open(os.path.join(IMG_PATH, self.img_list[img_index])).convert('RGB')
        monet_img = Image.open(os.path.join(MONET_PATH, self.monet_list[monet_index])).convert('RGB')
        
        img = self.tfm(img)
        monet_img = self.tfm(monet_img)
        
        img = (img - 0.5) * 2
        monet_img = (monet_img - 0.5) * 2
        
        return img, monet_img