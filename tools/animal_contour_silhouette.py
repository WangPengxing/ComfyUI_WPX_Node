from PIL import Image
import numpy as np
from .image_process import images_process
import random
import time
import cv2
from scipy.ndimage import distance_transform_edt


def invert_transparency(image):
    """
    反转图像的透明度
    :param image: 输入图像
    :return: 透明度反转后的图像
    """
    r, g, b, a = image.split()
    # 将透明度反转  
    inverted_a = Image.eval(a, lambda x: 255 - x)
    # 创建白色背景
    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
    # 将图像与白色背景进行合成
    composite_image = Image.composite(image, white_bg, inverted_a)
    return composite_image

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
    # 如果交集区域坐标小于0，则返回0
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

def check_overlap(box, placed_boxes, threshold):
    """
    使用网格分区检测重叠。
    :param box: 待检测的矩形框
    :param placed_boxes: 已放置的矩形框列表
    :param threshold: 重叠阈值
    :return: 布尔值，表示是否存在重叠
    """
    for placed_box in placed_boxes:
        iou = compute_iou(box, placed_box)
        if iou > threshold:
            return True
    return False

def place_stickers_inside_contour(base_image, mask, stickers, placed_boxes, max_sticker_count=200, overlap_threshold=0.5, bg_sticker_size=300):
    """
    在轮廓内部放置贴纸。
    :param base_image: 基础图像
    :param mask: 二值化掩码
    :param stickers: 贴纸列表
    :param placed_boxes: 已放置的矩形框列表
    :param max_sticker_count: 最大贴纸数量
    :param overlap_threshold: 重叠阈值
    :param bg_scale: 背景贴纸缩放系数
    :return: 结果图像和贴纸数量计数器
    """
    result_image = base_image.copy()
    height, width = mask.shape
    # 获取有效区域的边界
    rows, cols = np.where(mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return result_image
    min_y, max_y = np.min(rows), np.max(rows)
    min_x, max_x = np.min(cols), np.max(cols)
    # 预先计算和缓存变换后的贴纸
    transformed_stickers = []
    for sticker in stickers:
        angle = random.uniform(-5, 5)
        bg_scale = round(bg_sticker_size / max(sticker.size), 2)
        scale = bg_scale * random.uniform(0.95, 1.05)
        transformed = transform_sticker(sticker, angle, scale)
        if transformed:
            transformed_stickers.append(transformed)
    # 使用网格来加速位置选择
    grid_size = 30
    grid_positions = []
    for x in range(min_x, max_x, grid_size):
        for y in range(min_y, max_y, grid_size):
            if mask[y, x]:
                grid_positions.append((x, y))
    random.shuffle(grid_positions)  # 随机化网格位置
    counter = 0 # 贴纸计数器
    for x, y in grid_positions:
        if counter >= max_sticker_count:
            break
        # 从缓存中选择贴纸
        sticker = transformed_stickers[counter % len(transformed_stickers)]
        sticker_width, sticker_height = sticker.size
        top_left = (int(x - sticker_width/2), int(y - sticker_height/2))
        box = (top_left[0], top_left[1], top_left[0] + sticker_width, top_left[1] + sticker_height)
        # 检查是否超出边界
        if (box[0] < 0 or box[1] < 0 or 
            box[2] >= width or box[3] >= height):
            continue
        # 检查重叠
        if check_overlap(box, placed_boxes, overlap_threshold):
            continue
        # 放置贴纸
        result_image.paste(sticker, top_left, sticker)
        placed_boxes.append(box)
        counter += 1
    print(f"轮廓最大外接矩形内部放置贴纸数量累计：{counter}")
    return result_image

def find_visual_center(mask):
    """
    使用距离变换找到轮廓内的最佳视觉中心
    原理：找到距离轮廓边缘最远的点，该点更可能是视觉上的中心
    """
    dist_transform = distance_transform_edt(mask)   # 计算距离变换
    dist_transform = cv2.GaussianBlur(dist_transform, (7, 7), 0)    # 使用高斯模糊平滑距离图
    max_dist_loc = np.unravel_index(dist_transform.argmax(), dist_transform.shape)  # 找到距离最大的点
    return max_dist_loc[1], max_dist_loc[0]  # 返回 (x, y)

def find_safe_sticker_size(mask, center_x, center_y, initial_size):
    """
    找到在给定中心点位置可以放置的最大贴纸尺寸
    使用二分查找来优化搜索过程
    """
    left, right = 10, initial_size
    best_size = 10

    while left <= right:
        mid = (left + right) // 2
        radius = mid // 2
        # 创建临时圆形掩码来模拟贴纸
        temp_mask = np.zeros_like(mask)
        cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
        # 检查是否完全在轮廓内
        if np.all(mask[temp_mask == 255] == 255):
            best_size = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return best_size

def place_sticker_in_visual_center(transparent_image, main_sticker, bg_img):
    """
    在轮廓内放置主贴纸
    """
    # 将PIL图片转换为numpy数组
    animal_contour = np.array(transparent_image)
    if animal_contour.shape[2] != 4:
        raise ValueError("输入图片必须包含透明通道")
    # 获取 alpha 通道并创建掩码
    alpha = animal_contour [:, :, 3]
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    # 找到视觉中心
    center_x, center_y = find_visual_center(mask)
    # 计算初始最大可能尺寸
    initial_max_size = min(animal_contour.shape[1], animal_contour.shape[0])
    # 找到安全的贴纸尺寸
    safe_size = find_safe_sticker_size(mask, center_x, center_y, initial_max_size)
    if safe_size > 600:
        safe_size = 600
    random_scale = random.uniform(0.9, 1.0)
    safe_scale = int(safe_size * random_scale) / max(main_sticker.size)
    # 读取并调整贴纸大小
    main_sticker = main_sticker.resize((int(main_sticker.size[0] * safe_scale), int(main_sticker.size[1] * safe_scale)), Image.LANCZOS)
    # 计算贴纸放置位置
    x = center_x - main_sticker.width // 2
    y = center_y - main_sticker.height // 2
    bg_img.paste(main_sticker, (x, y), main_sticker)
    return bg_img

def animal_contour_silhouette_main(image_path, stickers_path, main_sticker_path, max_sticker_count=200, overlap_threshold=0.5, bg_sticker_size=250, crop_image=False, padding=0):
    """
    主要的贴图排列函数
    :param image_path: 动物图片路径
    :param stickers_path: 贴纸文件夹路径
    :param main_sticker_path: 主图贴纸路径
    :param max_sticker_count: 最大贴纸数量
    :param overlap_threshold: 重叠阈值
    :param bg_sticker_size: 背景贴纸大小
    :param crop_image: 是否进行正方形裁剪
    :param padding: 裁剪时额外添加的边距
    """
    # 加载贴图
    bg_stickers = images_process(stickers_path, crop_image=crop_image, padding=padding)
    main_sticker = images_process(main_sticker_path, crop_image=crop_image, padding=padding)[0].convert('RGBA')
    print(f"加载的贴图数量: {len(bg_stickers)}")
    if not bg_stickers:
        print("没有找到有效的贴图。")
        return
    # 加载透明背景图片
    transparent_image = images_process(image_path)[0].convert('RGBA')
    # 将黑白图的黑色区域改为透明
    data = transparent_image.getdata()  # 获取图像数据  
    transparent_image.putdata([(r, g, b, 0) if r == 0 and g == 0 and b == 0 else (r, g, b, a) for r, g, b, a in data])  # 将黑色区域改为透明
    # 缩放透明背景图片
    trans_img_scale = round(1600 / max(transparent_image.size), 2)
    transparent_image = transparent_image.resize((int(transparent_image.size[0] * trans_img_scale), int(transparent_image.size[1] * trans_img_scale)), Image.LANCZOS)
    # 创建与透明背景图片大小一致的画布
    convas = Image.new("RGBA", transparent_image.size, (255, 255, 255, 0))
    convas_mask = np.array(convas.convert('L')) // 255
    # 堆叠贴纸后的画布
    tiled_convas = place_stickers_inside_contour(convas, convas_mask, bg_stickers, [], max_sticker_count, overlap_threshold, bg_sticker_size)
    # 反转透明度并应用
    inverted_mask = invert_transparency(transparent_image)
    result = Image.composite(tiled_convas, Image.new('RGBA', tiled_convas.size, (255, 255, 255, 0)), inverted_mask)
    # 将主图贴纸放置在轮廓中心
    result = place_sticker_in_visual_center(transparent_image, main_sticker, result)
    # 返回结果
    return result

if __name__ == "__main__":
    start_time = time.time()
    # 设置路径
    stickers_path = "images/stickers_folder"  # 贴纸文件夹路径
    image_path = "images/test10.png"  # 动物图片路径
    main_sticker_path = "images/stickers_folder/散装_画板 1 副本 41.png"  # 主图贴纸路径
    output_path = "images/output/output109.png"  # 最终输出路径
    # 执行贴图排列
    result_img = animal_contour_silhouette_main(image_path, stickers_path, main_sticker_path)
    result_img.save(output_path)
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")

