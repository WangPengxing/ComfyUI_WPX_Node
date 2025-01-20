import time 
import numpy as np
import torch
from .tools import images_process
from .tools import animal_contour_main, rectangle_contour_main, circle_contour_main, penetrate_style_main, animal_contour_silhouette_main, split_single_image
import cv2
from PIL import Image

# 动物轮廓
class AnimalContour:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animal_image": ("IMAGE",),
                "stickers": ("IMAGE",),
                "main_sticker": ("IMAGE",),
                "max_sticker_count": ("INT", {"default": 130, "min": 1, "max": 1000, "step": 1,}),
                "spacing": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "overlap_threshold_contour": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01,}),
                "bg_sticker_size": ("INT", {"default": 220, "min": 1, "max": 1000, "step": 1,}),
                "main_sticker_size": ("INT", {"default": 380, "min": 1, "max": 1000, "step": 1,}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "animal_contour"
    CATEGORY = "WPX_Node/main_image"

    def animal_contour(self, animal_image, stickers, main_sticker, max_sticker_count, spacing, overlap_threshold_contour, bg_sticker_size, main_sticker_size):
        result_image, _ = animal_contour_main(
            image_path = animal_image,
            stickers_paths = stickers,
            main_sticker_path = main_sticker,
            max_sticker_count = max_sticker_count,
            spacing = spacing,
            overlap_threshold_contour = overlap_threshold_contour,
            bg_sticker_size = bg_sticker_size,
            main_sticker_size = main_sticker_size
        )
        # 将结果转换为归一化的numpy数组, dim:(H, W, C)
        result_np = np.array(result_image).astype(np.float32) / 255
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 添加批次维度
        result_np = np.expand_dims(result_np, axis=0)
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 转换为张量，dim:(B, H, W, C)
        img = torch.from_numpy(result_np)
        print(f"[DEBUG] img.shape: {img.shape}")
        return (img,)

# 动物轮廓剪影
class AnimalContourSilhouette:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animal_image": ("IMAGE",),
                "stickers": ("IMAGE",),
                "main_sticker": ("IMAGE",),
                "max_sticker_count": ("INT", {"default": 200, "min": 1, "max": 1000, "step": 1,}),
                "overlap_threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01,}),
                "bg_sticker_size": ("INT", {"default": 250, "min": 1, "max": 1000, "step": 1,}),
                "crop_image": ("BOOLEAN", {"default": False,}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,}),
            },
        }
    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("image",)
    FUNCTION = "animal_contour_silhouette"
    CATEGORY = "WPX_Node/main_image"
    def animal_contour_silhouette(self, animal_image, stickers, main_sticker, max_sticker_count, overlap_threshold, bg_sticker_size, crop_image, padding):
        result_image = animal_contour_silhouette_main(
            image_path = animal_image,
            stickers_path = stickers,
            main_sticker_path = main_sticker,
            max_sticker_count = max_sticker_count,
            overlap_threshold = overlap_threshold,
            bg_sticker_size = bg_sticker_size,
            crop_image = crop_image,
            padding = padding
        )
        # 将结果转换为归一化的numpy数组, dim:(H, W, C)
        result_np = np.array(result_image).astype(np.float32) / 255
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 添加批次维度
        result_np = np.expand_dims(result_np, axis=0)
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 转换为张量，dim:(B, H, W, C)
        img = torch.from_numpy(result_np)
        print(f"[DEBUG] img.shape: {img.shape}")
        return (img,)

# 矩形轮廓
class RectangleContour:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rectangle_image": ("IMAGE",),
                "stickers": ("IMAGE",),
                "main_sticker": ("IMAGE",),
                "n_scales": ("INT", {"default": 4, "min": 1, "max": 10}),
                "sticker_size": ("INT", {"default": 280, "min": 1, "max": 1000}),
                "main_sticker_size": ("INT", {"default": 450, "min": 1, "max": 1000}),
                "cell_size_factor": ("FLOAT", {"default": 0.45, "min": 0, "max": 1, "step": 0.01,}),
                "crop_image": ("BOOLEAN", {"default": False,}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rectangle_contour"
    CATEGORY = "WPX_Node/main_image"
    def rectangle_contour(self, rectangle_image, stickers, main_sticker, n_scales, sticker_size, main_sticker_size, cell_size_factor, crop_image, padding):
        result_image = rectangle_contour_main(
            image_path = rectangle_image,
            stickers_path = stickers,
            main_sticker_path = main_sticker,
            n_scales = n_scales,
            sticker_size = sticker_size, 
            main_sticker_size = main_sticker_size,
            cell_size_factor = cell_size_factor,
            crop_image = crop_image,
            padding = padding
        )
        # 将结果转换为归一化的numpy数组, dim:(H, W, C)
        result_np = np.array(result_image).astype(np.float32) / 255
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 添加批次维度
        result_np = np.expand_dims(result_np, axis=0)
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 转换为张量，dim:(B, H, W, C)
        img = torch.from_numpy(result_np)
        print(f"[DEBUG] img.shape: {img.shape}")
        return (img,)

# 圆形轮廓
class CircleContour:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "circle_image": ("IMAGE",),
                "stickers": ("IMAGE",),
                "main_sticker": ("IMAGE",),
                "n_scales": ("INT", {"default": 3, "min": 1, "max": 10}),
                "max_sticker_count": ("INT", {"default": 80, "min": 1, "max": 1000}),
                "spacing": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "overlap_threshold_contour": ("FLOAT", {"default": 0.60, "min": 0, "max": 1, "step": 0.01,}),
                "bg_sticker_size": ("INT", {"default": 330, "min": 1, "max": 1000}),
                "main_sticker_size": ("INT", {"default": 450, "min": 1, "max": 1000}),
                "crop_image": ("BOOLEAN", {"default": False,}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "circle_contour"
    CATEGORY = "WPX_Node/main_image"
    def circle_contour(self, circle_image, stickers, main_sticker, n_scales, max_sticker_count, spacing, overlap_threshold_contour, 
                       bg_sticker_size, main_sticker_size, crop_image, padding):
        result_image = circle_contour_main(
            image_path = circle_image,
            stickers_paths = stickers,
            main_sticker_path = main_sticker,
            n_scales = n_scales,
            max_sticker_count = max_sticker_count,
            spacing = spacing,
            overlap_threshold_contour = overlap_threshold_contour,
            bg_sticker_size = bg_sticker_size,
            main_sticker_size = main_sticker_size,
            crop_image = crop_image,
            padding = padding
        )
        # 将结果转换为归一化的numpy数组, dim:(H, W, C)
        result_np = np.array(result_image).astype(np.float32) / 255
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 添加批次维度
        result_np = np.expand_dims(result_np, axis=0)
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 转换为张量，dim:(B, H, W, C)
        img = torch.from_numpy(result_np)
        print(f"[DEBUG] img.shape: {img.shape}")
        return (img,)

# 渗透风格
class PenetrateStyle:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sticker_set_path": ("IMAGE",),
                "single_sticker_path": ("IMAGE",),
                "circumcircle_sticker_path": ("IMAGE",),    
                "main_sticker_path": ("IMAGE",),
                "bg_width": ("INT", {"default": 1600, "min": 1, "max": 10000}),
                "bg_height": ("INT", {"default": 1600, "min": 1, "max": 10000}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 10}),
                "cols": ("INT", {"default": 1, "min": 1, "max": 10}),
                "row1_style": (["style1", "style2", "style3", "style4", "style5"], {"default": "style1"}),
                "row1_y_offset": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "row2_style": (["style1", "style2", "style3", "style4", "style5"], {"default": "style2"}),
                "row2_y_offset": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "row3_style": (["style1", "style2", "style3", "style4", "style5"], {"default": "style3"}),
                "row3_y_offset": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "sticker_set_cols": ("INT", {"default": 4, "min": 1, "max": 10}),
                "random_stickers_rows": ("INT", {"default": 3, "min": 1, "max": 10}),
                "sticker_contour_num": ("INT", {"default": 4, "min": 1, "max": 10}),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "penetrate_style"
    CATEGORY = "WPX_Node/main_image"
    def penetrate_style(self, sticker_set_path, single_sticker_path, circumcircle_sticker_path, main_sticker_path, bg_width, bg_height, rows, cols, row1_style, row2_style, row3_style, 
        row1_y_offset, row2_y_offset, row3_y_offset, sticker_set_cols, random_stickers_rows, sticker_contour_num):
        result_image = penetrate_style_main(
            sticker_set_path = sticker_set_path,
            single_sticker_path = single_sticker_path,
            circumcircle_sticker_path = circumcircle_sticker_path,
            main_sticker_path = main_sticker_path,
            bg_width = bg_width,
            bg_height = bg_height,
            rows = rows,
            cols = cols,
            styles = (row1_style, row2_style, row3_style),
            y_cell_offset = [row1_y_offset, row2_y_offset, row3_y_offset],
            sticker_set_cols = sticker_set_cols,
            random_stickers_rows = random_stickers_rows,
            sticker_contour_num = sticker_contour_num
        )
        # 将结果转换为归一化的numpy数组, dim:(H, W, C)
        result_np = np.array(result_image).astype(np.float32) / 255
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 添加批次维度
        result_np = np.expand_dims(result_np, axis=0)
        print(f"[DEBUG] result_np.shape: {result_np.shape}")
        # 转换为张量，dim:(B, H, W, C)
        img = torch.from_numpy(result_np)
        print(f"[DEBUG] img.shape: {img.shape}")
        return (img,)

# 分割贴纸
class SplitStickers:
    def __init__(self):
        pass    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image_path": ("IMAGE",)},
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "split_stickers"
    CATEGORY = "WPX_Node/main_image"

    OUTPUT_IS_LIST = (True,)
    def split_stickers(self, image_path):
        result_images = split_single_image(image_path)
        for i, result_image in enumerate(result_images):
            # 将结果转换为归一化的numpy数组, dim:(H, W, C)
            result_np = np.array(result_image).astype(np.float32) / 255
            # 添加批次维度
            result_np = np.expand_dims(result_np, axis=0)
            # 转换为张量，dim:(B, H, W, C)
            img = torch.from_numpy(result_np)
            # 作为列表返回
            result_images[i] = img
        return (result_images,)

# 判断行列数
class DetermineRowsAndCols:
    
     def __init__(self):
         pass
     
     @classmethod
     def INPUT_TYPES(cls):
         return {
             "required": {
                 "num1": ("INT", {"default": 2, "min": 1, "max": 10}),
                 "num2": ("INT", {"default": 2, "min": 1, "max": 10}),
                 "total_count": ("INT", {"default": 10, "min": 1, "max": 50})
             }
         }
     
     RETURN_TYPES = ("INT", "INT")
     RETURN_NAMES = ("Rows", "Cols")
     FUNCTION = "determine_rows_and_cols"
     CATEGORY = "WPX_Node"

     def determine_rows_and_cols(self, num1, num2, total_count):
         
         min_num = min(num1, num2)
         max_num = max(num1, num2)

         if min_num*min_num == total_count:
             return (min_num, min_num)
         elif min_num*min_num < total_count and max_num*min_num >= total_count:
             return (max_num, min_num)
         else:
             return (max_num, max_num)


def _split_mask_elements(image, mask, padding=10, filtration_area=0.0025):
    """根据mask分割图像元素"""
    # 将PIL图像转换为numpy数组
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # 确保mask是单通道二值图像
    if len(mask_np.shape) > 2:
        mask_np = mask_np[:, :, 0] if mask_np.shape[-1] > 1 else mask_np.squeeze()
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排除轮廓面积占比小于filtration_area的轮廓
    contours = [contour for contour in contours if cv2.contourArea(contour) > filtration_area * image_np.shape[0] * image_np.shape[1]]
    result_images = []
    
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 添加padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_np.shape[1] - x, w + 2 * padding)
        h = min(image_np.shape[0] - y, h + 2 * padding)
        
        # 创建透明背景
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 复制RGB通道和原始alpha通道
        if image_np.shape[-1] == 4:  # 如果原图有alpha通道
            rgba[:, :, :4] = image_np[y:y+h, x:x+w, :4]
        else:  # 如果原图只有RGB通道
            rgba[:, :, :3] = image_np[y:y+h, x:x+w, :3]
        
        # 创建当前轮廓的mask
        contour_mask = np.zeros_like(mask_np)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # 只取当前轮廓内的区域作为alpha通道
        alpha = contour_mask[y:y+h, x:x+w]
        
        # 如果原图有alpha通道，需要将原始alpha和轮廓mask结合
        if image_np.shape[-1] == 4:
            original_alpha = image_np[y:y+h, x:x+w, 3]
            rgba[:, :, 3] = cv2.bitwise_and(original_alpha, alpha)
        else:
            rgba[:, :, 3] = alpha
        
        # 将numpy数组转换为PIL图像
        result_image = Image.fromarray(rgba)
        result_image.show()
        result_images.append(result_image)
    
    return result_images

# 分割mask元素
class SplitMaskElements:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "filtration_area": ("FLOAT", {"default": 0.0025, "min": 0, "max": 1, "step": 0.0001}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_mask_elements"
    CATEGORY = "WPX_Node"

    def split_mask_elements(self, image, mask, padding=10, filtration_area=0.0025):
        image = images_process(image)[0]
        mask = images_process(mask)[0]
        # 分割mask元素
        result_images = _split_mask_elements(image, mask, padding, filtration_area)
        
        # 处理结果图像
        processed_images = []
        for result_image in result_images:
            # 将PIL图像转换为numpy数组并归一化
            result_np = np.array(result_image).astype(np.float32) / 255
            # 添加批次维度
            result_np = np.expand_dims(result_np, axis=0)
            # 转换为张量
            img = torch.from_numpy(result_np)
            processed_images.append(img)
            
        return (processed_images,)


if __name__ == "__main__":
    # animal_contour()
    # print("动物轮廓完成" + "-"*100)
    # rectangle_contour()
    # print("矩形轮廓完成" + "-"*100)
    # circle_contour()
    # print("圆形轮廓完成" + "-"*100)
    # penetrate_style()
    generator = SplitMaskElements()
    image = Image.open("D:/test_image/input/02.png")
    mask = Image.open("D:/test_image/input/02_mask.png")
    result_images = generator.split_mask_elements(image, mask)
    print("分割mask元素完成" + "-"*100)
