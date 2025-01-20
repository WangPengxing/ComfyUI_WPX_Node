import cv2
import numpy as np
from PIL import Image
import random
import glob
import os
from rtree import index
import time
from .image_process import images_process


def extract_contour(mask):
    """
    提取二值化mask的轮廓和最大外接矩形。
    :param mask: 二值化图像 (numpy 数组)
    :return: 轮廓列表和最大外接矩形 (x,y,w,h)
    """
    # 将mask转换为uint8类型
    mask_uint8 = (mask * 255).astype(np.uint8)
    # 提取轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 如果找到轮廓
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.boundingRect(largest_contour)
        return contours, rect
    return contours, None

def compute_iou(box1, box2):
    """
    计算两个矩形框的IoU。
    :param box1: 第一个矩形框坐标 (x1,y1,x2,y2)
    :param box2: 第二个矩形框坐标 (x1,y1,x2,y2)
    :return: IoU值
    """
    # 计算交集区域坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    # 计算交集面积
    intersection = (x2 - x1) * (y2 - y1)
    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 计算并集面积
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def is_overlap_allowed(box1, box2, threshold):
    """
    判断两个框的重叠是否在阈值范围内。
    :param box1: 第一个矩形框坐标 (x1,y1,x2,y2)
    :param box2: 第二个矩形框坐标 (x1,y1,x2,y2)
    :param threshold: 重叠阈值
    :return: 布尔值，表示是否允许重叠
    """
    iou = compute_iou(box1, box2)
    return iou <= threshold

def transform_sticker(sticker, angle, scale=1.0):
    """
    旋转和缩放贴纸。
    :param sticker: 输入贴纸图像
    :param angle: 旋转角度
    :param scale: 缩放比例
    :return: 变换后的贴纸图像
    """
    try:
        # 缩放贴纸
        if scale != 1.0:
            new_size = tuple(int(dim * scale) for dim in sticker.size)
            sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
            
        # 旋转贴纸
        if angle != 0:
            sticker = sticker.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
            
        return sticker
    except Exception as e:
        print(f"Error transforming sticker: {e}")
        return None

def compute_contour_direction(contour, point, window_size=5):
    """
    计算轮廓上某点的切线方向。
    :param contour: 轮廓点集
    :param point: 目标点坐标 (x,y)
    :param window_size: 计算切线的窗口大小
    :return: 切线角度(度)
    """
    # 找到点在轮廓上的索引
    indices = np.where((contour[:,0,0] == point[0]) & (contour[:,0,1] == point[1]))[0]
    if len(indices) == 0:
        return 0
    idx = indices[0]
    n = len(contour)
    
    # 获取窗口范围内的点
    start = max(idx - window_size, 0)
    end = min(idx + window_size, n - 1)
    pts = contour[start:end]
    if len(pts) < 2:
        return 0
        
    # 使用最小二乘法拟合直线
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(vy, vx))
    return angle

def create_spatial_index(boxes):
    """
    创建空间索引。
    :param boxes: 矩形框列表
    :return: 空间索引对象
    """
    p = index.Property()
    idx = index.Index(properties=p)
    for i, box in enumerate(boxes):
        idx.insert(i, box)
    return idx

def check_overlap_with_spatial_index(box, spatial_idx, boxes, threshold):
    """
    使用空间索引进行重叠检测。
    :param box: 待检测的矩形框
    :param spatial_idx: 空间索引对象
    :param boxes: 已放置的矩形框列表
    :param threshold: 重叠阈值
    :return: 布尔值，表示是否存在重叠
    """
    query_bbox = box # 查询框
    possible_matches = list(spatial_idx.intersection(query_bbox)) # 可能匹配的索引
    for idx in possible_matches:
        if not is_overlap_allowed(box, boxes[idx], threshold):
            return True
    return False

def sample_contour_points(contour, spacing=20):
    """
    在轮廓上均匀采样点。
    :param contour: 轮廓点集
    :param spacing: 采样间隔（像素）
    :return: 采样点列表
    """
    sampled_points = []
    contour_length = cv2.arcLength(contour, True)
    num_samples = int(contour_length // spacing)
    
    # 累计弧长采样
    accumulated_length = 0
    target_length = spacing
    # 遍历轮廓点
    for i in range(1, len(contour)):
        pt1 = contour[i-1][0]   # 当前点
        pt2 = contour[i][0] # 下一个点
        segment_length = np.linalg.norm(pt2 - pt1)  # 线段长度
        if accumulated_length + segment_length >= target_length:
            ratio = (target_length - accumulated_length) / segment_length  # 比例
            x = pt1[0] + ratio * (pt2[0] - pt1[0])  # 采样点x坐标    
            y = pt1[1] + ratio * (pt2[1] - pt1[1])  # 采样点y坐标
            sampled_points.append((int(x), int(y)))  # 采样点
            target_length += spacing # 更新目标长度
        accumulated_length += segment_length # 更新累计长度
        if len(sampled_points) >= num_samples:
            break
    return sampled_points

def place_stickers_along_contour(base_image, contour, stickers, max_sticker_count=500, spacing=20, overlap_threshold=0.5, bg_scale=1.0, sticker_index=0):
    """
    沿轮廓放置贴纸。
    :param base_image: 基础图像
    :param contour: 轮廓点集
    :param stickers: 贴纸列表
    :param max_sticker_count: 最大贴纸数量
    :param spacing: 贴纸间隔
    :param overlap_threshold: 重叠阈值
    :param bg_scale: 背景贴纸缩放系数
    :param sticker_index: 贴纸下标索引
    :return: 结果图像和已放置的矩形框列表
    """
    result_image = base_image.copy()
    sampled_points = sample_contour_points(contour, spacing)
    placed_boxes = []
    sticker_count = 0
    
    for idx, point in enumerate(sampled_points):
        if len(placed_boxes) >= max_sticker_count:
            break
            
        angle = compute_contour_direction(contour, point)
        sticker = stickers[sticker_index % len(stickers)]
        scale = bg_scale
        
        transformed_sticker = transform_sticker(sticker, angle, scale)
        if transformed_sticker is None:
            continue

        sticker_width, sticker_height = transformed_sticker.size
        top_left = (int(point[0] - sticker_width/2), int(point[1] - sticker_height/2))
        box = (top_left[0], top_left[1], top_left[0] + sticker_width, top_left[1] + sticker_height)

        # 检查重叠
        spatial_idx = create_spatial_index(placed_boxes)
        if not check_overlap_with_spatial_index(box, spatial_idx, placed_boxes, overlap_threshold):
            if top_left[0] < 0:
                top_left = (0, top_left[1])
            if top_left[1] < 0:
                top_left = (top_left[0], 0)
            if top_left[0] + sticker_width > result_image.width:
                top_left = (result_image.width - sticker_width, top_left[1])
            if top_left[1] + sticker_height > result_image.height:
                top_left = (top_left[0], result_image.height - sticker_height)
            print(f"贴纸位置：{top_left}")
            result_image.paste(transformed_sticker, top_left, transformed_sticker)
            placed_boxes.append(box)
            sticker_index += 1
            sticker_count += 1
                
    print(f"沿轮廓放置贴纸数量：{sticker_count}")
    return result_image, placed_boxes, sticker_index

def place_stickers_inside_contour(base_image, mask, stickers, placed_boxes, max_sticker_count=500, overlap_threshold=0.5, bg_scale=1.0, sticker_index = 0):
    """
    在轮廓内部放置贴纸。
    :param base_image: 基础图像
    :param mask: 二值化掩码
    :param stickers: 贴纸列表
    :param placed_boxes: 已放置的矩形框列表
    :param max_sticker_count: 最大贴纸数量
    :param overlap_threshold: 重叠阈值
    :param bg_scale: 背景贴纸缩放系数
    :param sticker_index: 贴纸下标索引
    :return: 结果图像和贴纸下标索引
    """
    result_image = base_image.copy()
    height, width = mask.shape
    
    # 获取有效区域的边界
    rows, cols = np.where(mask > 0) 
    if len(rows) == 0 or len(cols) == 0:
        return result_image, sticker_index
        
    min_y, max_y = np.min(rows), np.max(rows)
    min_x, max_x = np.min(cols), np.max(cols)
    
    # 贴纸数量计数器
    sticker_count = 0
    attempts = 0
    max_attempts = max_sticker_count * 10
    
    while sticker_count < max_sticker_count and attempts < max_attempts:
        attempts += 1
        
        # 随机选择位置和贴纸参数
        x = random.randint(0, (max_x-min_x)//20)*20 + min_x
        y = random.randint(0, (max_y-min_y)//20)*20 + min_y
        
        if not mask[y, x]:
            continue
            
        # sticker = random.choice(stickers)
        sticker = stickers[sticker_index % len(stickers)]
        angle = random.uniform(-10, 10)
        scale = bg_scale * random.uniform(0.95, 1.05)
        
        # 变换贴纸
        transformed_sticker = transform_sticker(sticker, angle, scale)
        if transformed_sticker is None:
            continue
            
        # 计算贴纸位置和边界框
        sticker_width, sticker_height = transformed_sticker.size
        top_left = (int(x - sticker_width/2), int(y - sticker_height/2))
        box = (top_left[0], top_left[1], top_left[0] + sticker_width, top_left[1] + sticker_height)
        
        # 检查是否超出边界
        if (box[0] < 0 or box[1] < 0 or 
            box[2] >= width or box[3] >= height):
            continue
            
        # 检查重叠
        overlap = False
        spatial_idx = create_spatial_index(placed_boxes)
        if check_overlap_with_spatial_index(box, spatial_idx, placed_boxes, overlap_threshold):
            continue
            
        # 放置贴纸
        result_image.paste(transformed_sticker, top_left, transformed_sticker)
        placed_boxes.append(box)
        sticker_index += 1
        sticker_count += 1 
    print(f"内部放置贴纸数量累计：{sticker_count}")
    return result_image, sticker_index

def place_main_sticker(base_image, main_sticker, rect, scale=0.4):
    """
    在最大外接矩形中心放置主图贴纸。
    :param base_image: 基础图像
    :param main_sticker: 主图贴纸
    :param rect: 外接矩形 (x,y,w,h)
    :param scale: 缩放比例
    :return: 结果图像
    """
    result_image = base_image.copy()
    if main_sticker is None:
        return result_image
        
    # 计算矩形中心
    x, y, w, h = rect
    center_x = x + w//2
    center_y = y + h//2
    
    # 缩放主图贴纸
    main_w, main_h = main_sticker.size
    new_w = int(main_w * scale)
    new_h = int(main_h * scale)
    main_sticker = main_sticker.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 计算粘贴位置
    paste_x = center_x - new_w//2
    paste_y = center_y - new_h//2
    
    result_image.paste(main_sticker, (paste_x, paste_y), main_sticker)
    return result_image

def animal_contour_main(image_path, stickers_paths, main_sticker_path, max_sticker_count = 500, spacing = 20,
          overlap_threshold_contour = 0.5, bg_sticker_size = 220, main_sticker_size = 380, sticker_index = 0,
          crop_image = False, padding = 0):
    """
    主函数，处理图像并放置贴纸。
    :param image_path: 输入图像路径
    :param stickers_paths: 贴纸文件夹路径
    :param main_sticker_path: 主图贴纸路径
    :param max_sticker_count: 最大贴纸数量
    :param spacing: 轮廓采样间隔
    :param overlap_threshold_contour: 贴纸轮廓重叠阈值
    :param bg_sticker_size: 背景贴纸尺寸
    :param main_sticker_size: 主图贴纸尺寸
    :param sticker_index: 贴纸下标索引
    :return: 结果图像和贴纸下标索引
    """
    # 加载图像和贴纸
    # base_image, main_sticker, stickers = load_images(image_path, stickers_paths, main_sticker_path)
    base_image = images_process(image_path)
    base_image = base_image[0]  # 取第一张传入的动物轮廓图片
    main_sticker = images_process(main_sticker_path, crop_image=crop_image, padding=padding)
    main_sticker = main_sticker[0]  # 取第一张传入的主图贴纸
    stickers = images_process(stickers_paths, crop_image=crop_image, padding=padding)
    transparent_bg = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    # 确定背景贴纸、主图贴纸按长边的缩放系数，保留小数点后两位    
    bg_scale = round(bg_sticker_size / max(stickers[0].size), 2)
    main_scale = round(main_sticker_size / max(main_sticker.size), 2)
    # print(f"背景贴纸缩放系数：{bg_scale}, 主图贴纸缩放系数：{main_scale}")

    # 提取轮廓
    mask = np.array(base_image.convert('L')) // 255
    contours, rect = extract_contour(mask)
    if not contours:
        print("未找到任何轮廓。")
        return transparent_bg, sticker_index
    contour = max(contours, key=cv2.contourArea)

    # 启用性能分析
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # 放置轮廓贴纸
    result_image, placed_boxes, sticker_index = place_stickers_along_contour(
        transparent_bg, contour, stickers, max_sticker_count, spacing, overlap_threshold_contour, bg_scale, sticker_index
    )

    # 放置内部贴纸
    result_image, sticker_index = place_stickers_inside_contour(
        result_image, mask, stickers, placed_boxes, max_sticker_count, overlap_threshold_contour, bg_scale, sticker_index
    )
    
    # 放置主图贴纸
    result_image = place_main_sticker(result_image, main_sticker, rect, main_scale)
    # 停止性能分析
    # profiler.disable()
    # profiler.print_stats()
    return result_image, sticker_index

if __name__ == "__main__":
    # 开始时间
    start_time = time.time()
    # 获取项目的根目录  
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 将路径中的\替换为/
    root = root.replace("\\", "/") + "/"

    image_path = os.path.join(root, "images/test15.png")  # 动物图片路径
    stickers_paths = os.path.join(root, "images/baby_bear/all")  # 贴纸文件夹路径
    main_sticker_path = os.path.join(root, "images/baby_bear/all/图层 2.png")  # 主图贴纸路径
    output_path = os.path.join(root, "images/output/output99.png")  # 输出图像路径

    result_image, _ = animal_contour_main(
        image_path = image_path,
        stickers_paths = stickers_paths,
        main_sticker_path = main_sticker_path,
        max_sticker_count = 130,
        spacing = 10,
        overlap_threshold_contour = 0.6,
        bg_sticker_size = 220,
        main_sticker_size = 380,
        sticker_index = 0
    )
    # 保存结果
    try:
        result_image.save(output_path, format='PNG')
        print(f"贴纸放置完成，结果保存在 {output_path}")
    except Exception as e:
        print(f"Error saving output image {output_path}: {e}")

    # 结束时间
    end_time = time.time()
    print(f"总时间：{end_time - start_time:.2f}秒")
