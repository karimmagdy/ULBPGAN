from skimage.color import rgb2gray, rgba2rgb
from util.util import tensor2im 
import numpy as np
import torch

class LBP_Extractor():

  def __init__(self, opt):
    super().__init__()
    self.opt = opt
    self.output_nc = opt.output_nc
    
  def get_pixel(img, center, x, y): 

      new_value = 0

      try: 
          # If local neighbourhood pixel  
          # value is greater than or equal 
          # to center pixel values then  
          # set it to 1 
          if img[x][y] >= center: 
              new_value = 1

      except: 
          # Exception is required when  
          # neighbourhood value of a center 
          # pixel value is null i.e. values 
          # present at boundaries. 
          pass

      return new_value 

  # Function for calculating LBP 
  def lbp_calculated_pixel(img, x, y): 
      # print((img[x][y]).shape)
      center = img[x][y] 

      val_ar = [] 

      # top_left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y-1)) 

      # top 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y)) 

      # top_right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x-1, y + 1)) 

      # right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x, y + 1)) 

      # bottom_right 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y + 1)) 

      # bottom 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y)) 

      # bottom_left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x + 1, y-1)) 

      # left 
      val_ar.append(LBP_Extractor.get_pixel(img, center, x, y-1)) 

      # Now, we need to convert binary 
      # values to decimal 
      power_val = [1, 2, 4, 8, 16, 32, 64, 128] 

      val = 0

      for i in range(len(val_ar)): 
          val += val_ar[i] * power_val[i] 

      return val 

  def toLBP(tensor_bgr, is_tensor = True, output_nc = 1, back2tensor = True):
    
    # print('tensor_bgr tensor: ', tensor_bgr.shape)
    img_bgr = tensor2im(tensor_bgr)
    if is_tensor == True:
      #tensor_bgr = torch.unsqueeze(tensor_bgr, 0)
      
      # print('img_bgr tensor: ', img_bgr.shape)
      _,height, width, _ = img_bgr.shape
      
    else:
      #img_bgr = tensor_bgr
      # print('img_bgr non-tensor: ', img_bgr.shape)
      _, height, width = img_bgr.shape
    #print('img_bgr:', img_bgr.shape)
    # grayscale = rgb2gray(rgba2rgb(img_bgr))
    grayscale = rgb2gray(img_bgr)
    # print('grayscale:', grayscale.shape)
    grayscale = grayscale.reshape(height, width)
    # print('grayscale after reshape:', grayscale.shape)
    # print(grayscale.size)
    # print(height)
    # print(width)
    img_lbp = np.zeros((height, width), 
                      np.uint8)
    # print('img_lbp:', img_lbp.shape)                  
    for i in range(0, height): 
      for j in range(0, width): 
        img_lbp[i, j] = LBP_Extractor.lbp_calculated_pixel(grayscale, i, j)

    if(back2tensor):
      tensor = torch.tensor(img_lbp)
      tensor_cuda = tensor.to('cuda')
      tensor_full = torch.reshape(tensor_cuda, (1, 1, height, width))
      return tensor_full
    return img_lbp