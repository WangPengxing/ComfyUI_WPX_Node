import cv2
import numpy as np
import os
from pathlib import Path
import logging
from .image_process import images_process
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_output_folder(output_path):
    """创建输出文件夹"""
    try:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建输出文件夹失败: {str(e)}")
        return False

def create_levels_lut(black_point, white_point, midtones):
    """创建查找表（模拟色阶调整）"""
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < black_point:
            lut[i] = 0
        elif i > white_point:
            lut[i] = 255
        else:
            lut[i] = np.clip(int(255 * ((i - black_point) / (white_point - black_point)) ** midtones), 0, 255)
    return lut


def preprocess_image(image, canny_low=10, canny_high=200, blur_size=5, dilate_iter=2):
    """图像预处理"""
    try:
        lut = create_levels_lut(black_point=150, white_point=250, midtones=1.0)
        result = cv2.LUT(image, lut)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=dilate_iter) # 膨胀
        return dilated
    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        return None

def calculate_overlap_ratio(rect1, rect2):
    """计算两个矩形的交集面积比例"""
    try:
        # rect格式: (x, y, w, h)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # 计算交集矩形的坐标
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # 如果没有交集，返回0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        # 计算第二个矩形（附属元素）的面积
        area_rect2 = w2 * h2
        
        # 返回交集面积与附属元素面积的比例
        return intersection_area / area_rect2
    except Exception as e:
        logging.error(f"计算重叠比例失败: {str(e)}")
        return 0.0

def merge_related_contours(contours, overlap_threshold=0.2):
    """合并相关联的轮廓"""
    try:
        if not contours:
            return []
            
        # 按面积降序排序轮廓
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        merged_contours = []
        used_indices = set()
        
        # 遍历每个主要轮廓
        for i, main_contour in enumerate(sorted_contours):
            if i in used_indices:
                continue
                
            main_rect = cv2.boundingRect(main_contour)
            current_group = [main_contour]
            
            # 查找相关的附属元素
            for j, sub_contour in enumerate(sorted_contours):
                if j == i or j in used_indices:
                    continue
                    
                sub_rect = cv2.boundingRect(sub_contour)
                overlap_ratio = calculate_overlap_ratio(main_rect, sub_rect)
                
                # 如果重叠比例超过阈值，将附属元素添加到当前组
                if overlap_ratio > overlap_threshold:
                    current_group.append(sub_contour)
                    used_indices.add(j)
            
            # 合并当前组的所有轮廓
            merged_contour = np.concatenate(current_group)
            # 计算凸包
            hull = cv2.convexHull(merged_contour)
            merged_contours.append(hull)
            used_indices.add(i)
            
        return merged_contours
    except Exception as e:
        logging.error(f"合并相关轮廓失败: {str(e)}")
        return []

def find_sticker_contours(thresh):
    """查找贴纸轮廓"""
    try:
        # 查找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 过滤小轮廓
        min_area = 200
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # 合并相关联的轮廓
        merged_contours = merge_related_contours(valid_contours)
        return merged_contours
    except Exception as e:
        logging.error(f"查找贴纸轮廓失败: {str(e)}")
        return []

def extract_sticker(image, contour, margin=20):
    """提取单个贴纸"""
    try:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 添加边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        # 裁剪贴纸
        sticker = image[y:y+h, x:x+w]
        return sticker
    except Exception as e:
        logging.error(f"提取贴纸失败: {str(e)}")
        return None

def split_single_image(input_path):
    """处理单张图片"""
    try:
        # 读取图片
        images = images_process(input_path)
        image = images[0]
        # 将PIL图像转换为OpenCV格式
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 预处理图片
        thresh = preprocess_image(image)
        cv2.imshow("thresh", thresh)
        if thresh is None:
            return []

        # 查找贴纸轮廓
        contours = find_sticker_contours(thresh)
        if not contours:
            logging.warning("未找到贴纸")
            return []

        # 提取贴纸并加入列表
        stickers = []
        for i, contour in enumerate(contours):
            # 排除过小的轮廓 
            if cv2.contourArea(contour) < 128*128:
                continue
            sticker = extract_sticker(image, contour)
            # print(f"[DEBUG] sticker.shape: {sticker.shape}")
            sticker = cv2.cvtColor(sticker, cv2.COLOR_BGR2RGB)
            # 将贴纸转换为PIL图像
            sticker = Image.fromarray(sticker)
            sticker.show()

            if sticker is not None:
                stickers.append(sticker)
        return stickers
    except Exception as e:
        logging.error(f"处理图片失败: {str(e)}")
        return []

def split_images_in_folder(input_folder, output_folder):
    """处理文件夹中的所有图片"""
    try:
        # 检查输入文件夹是否存在
        if not os.path.exists(input_folder):
            raise ValueError("输入文件夹不存在")

        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        
        # 处理每张图片
        success_count = 0
        total_count = 0
        
        for file in os.listdir(input_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                total_count += 1
                input_path = os.path.join(input_folder, file)
                # output_path = os.path.join(output_folder, f"split_{Path(file).stem}")
                output_path = output_folder
                if split_single_image(input_path, output_path):
                    success_count += 1

        logging.info(f"处理完成: 成功 {success_count}/{total_count} 张图片")
        return True
    except Exception as e:
        logging.error(f"处理文件夹失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 使用示例
    input_path = "images/02.png"  # 替换为实际的输入图片路径
    output_path = "output1"     # 替换为实际的输出文件夹路径
    
    # 处理单张图片
    split_single_image(input_path)
    
    # 处理文件夹
    # split_images_in_folder(input_path, output_path)
