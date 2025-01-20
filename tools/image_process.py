import numpy as np
from PIL import Image
import os
import glob
import torch

def images_process(images, crop_image=False, padding=0):
    """
    处理图片，返回处理后的图片
    :param images: 图片路径or图片张量orPIL.Image.Image
    :param crop_image: 是否进行正方形裁剪，默认False
    :param padding: 裁剪时额外添加的边距，默认0
    :return: 处理后的图片列表
    """
    images_list = []
    if isinstance(images, str):
        # 如果传入的是文件夹路径，则遍历文件夹中的所有图片
        if os.path.isdir(images):
            images_files = glob.glob(os.path.join(images, '*')) # 图片路径列表
            images_list = [Image.open(image).convert("RGBA") for image in images_files] # PIL.Image图片列表  
        else:
            images_list.append(Image.open(images).convert("RGBA"))
    elif isinstance(images, torch.Tensor):
        # 将传入的张量转置为(B, H, W, C)
        if images.shape[1] == 3 or images.shape[1] == 4:
            images = images.permute(0, 2, 3, 1) # (B, H, W, C)
        
        print("[DEBUG]传入的是张量，张量的形状, ", images.shape)
        images_tensor = [images[i:i+1] for i in range(images.shape[0])]  # 将每个批次作为独立的图片
        # 遍历每张贴纸
        for img_tensor in images_tensor:
            # 假设输入张量是 (1, H, W, C) 格式，取第一维的1（即去掉批次维度）
            img_tensor = img_tensor.squeeze(0)  # 去掉批次维度，得到(C, H, W)格式
            # 将张量转换为PIL图像
            img_np = img_tensor.cpu().numpy()  # 将张量转换为numpy数组
            img_np = (img_np * 255).astype(np.uint8)
            # 将numpy数组转换为图像
            img = Image.fromarray(img_np)
            # 加入列表容器
            images_list.append(img)
    elif isinstance(images, list):
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGBA")
            images_list.append(image)
    elif isinstance(images, Image.Image):
        images_list.append(images.convert("RGBA"))
    else:
        print("传入的参数类型错误，", type(images))

    # 在返回之前添加裁剪处理
    if crop_image:
        cropped_images = []
        for img in images_list:
            # 获取图片的alpha通道
            alpha = img.split()[-1]
            # 获取非透明区域的边界框
            bbox = alpha.getbbox()
            if bbox:
                left, top, right, bottom = bbox
                # 计算边界框的宽度和高度
                width = right - left
                height = bottom - top
                # 使用最大边长构建正方形
                max_side = max(width, height) + 2 * padding
                
                # 计算原图中心点
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                # 计算正方形裁剪区域
                new_left = center_x - max_side // 2
                new_top = center_y - max_side // 2
                new_right = new_left + max_side
                new_bottom = new_top + max_side
                
                # 确保裁剪区域不超出图片范围
                img_width, img_height = img.size
                new_left = max(0, new_left)
                new_top = max(0, new_top)
                new_right = min(img_width, new_right)
                new_bottom = min(img_height, new_bottom)
                
                # 裁剪图片
                cropped_img = img.crop((new_left, new_top, new_right, new_bottom))
                cropped_images.append(cropped_img)
            else:
                # 如果没有非透明区域，直接添加原图
                cropped_images.append(img)
        return cropped_images
    
    return images_list
