import numpy as np
import cv2
import os

try:
    from models import BiSeNet
except ImportError:
    from .models import BiSeNet

import torch
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image


def vis_parsing_maps(parsing_anno, stride, save_im=False, save_dir='./res/test_res', image_path="1.png"):
    save_path = os.path.join(save_dir, image_path.split('/')[-1])

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        
        index = np.where(vis_parsing_anno == pi)

        # get only the face mask
        if pi >= 1 and pi <= 13 and pi in np.unique(vis_parsing_anno):
            vis_parsing_anno_color[index[0], index[1], :] = [255,255,255]

        else:
            vis_parsing_anno_color[index[0], index[1], :] = [0,0,0]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    if save_im:
        cv2.imwrite(save_path.replace(".jpg", "_mask.png"), vis_parsing_anno_color)


def evaluate(image_path, cp, respth='./res/test_res'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiSeNet(n_classes=n_classes)
    net.to(device)  # Move model to the correct device
    
    # Load the model weights on the correct device
    net.load_state_dict(torch.load(cp, map_location=device))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        print("Parsing: ", image_path)
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)  # Move input image to the correct device
        out = net(img)


        # parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        # vis_parsing_maps(parsing, stride=1, save_im=True, save_dir=respth, image_path=image_path)

class FaceParser:
    def __init__(self, weight, device):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        # Load the model to CPU or GPU depending on available device
        self.net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
        self.net.to(device)
        self.net.eval()
        self.device = device

        self.to_tensor = transforms.Compose([
            transforms.Resize((512, 512), Image.BILINEAR),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    @torch.no_grad()
    def inference(self, img):
        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255)
        
        img = self.to_tensor(img)
        img = img.to(self.device)
        out1, out2, out3 = self.net(img)

        # concatenate all elements of the out tuple in out
        # out = torch.cat((out1, out2, out3), dim=1)       

        # parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        # vis_parsing_maps(parsing, stride=1, save_im=True, save_dir="./face_parser/res/test_res")

        return out1
    
    # created so that no_grad is not there
    def segmentation_embedding(self, img):
        img = torch.clamp((img + 1) * 0.5, 0, 1)
        img = self.to_tensor(img)
        img = img.to(self.device)
        out1, out2, out3 = self.net(img)

        # concatenate all elements of the out tuple in out
        # out = torch.cat((out1, out2, out3), dim=1)      
        return out1



# if __name__ == "__main__":

#     for img_path in os.listdir("/media/data/dwij22/TS/data/Celeba"):
#         img_path = "/media/data/dwij22/TS/data/Celeba/" + img_path
#         evaluate(img_path, 'face_parser/79999_iter.pth', respth='/media/data/dwij22/TS/data/Celeba-masks')
#         break