import cv2
import numpy as np
import glob
from PIL import Image
import os
import random
import time
from .image_process import images_process


def load_images(image_path, stickers_path):
    """
    加载矩形图片和贴纸图片。
    :param image_path: 矩形图片路径
    :param stickers_path: 贴纸图片文件夹路径
    :return: 矩形图片（RGBA格式）、贴纸列表
    """
    shape_img = Image.open(image_path).convert("RGBA")
    sticker_files = glob.glob(os.path.join(stickers_path, '*'))
    stickers = [Image.open(sticker).convert("RGBA") for sticker in sticker_files]
    return shape_img, stickers

def detect_rectangle(shape_image):
    """
    检测矩形，并返回左上坐标和宽高。
    :param shape_image: 输入图片
    :return: 左上坐标和宽高
    """
    # 转换为灰度图并二值化
    gray_img = np.array(shape_image.convert("L"))
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 查找最大轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    # 获取矩形参数
    x, y, w, h = cv2.boundingRect(cnt)
    print("矩形参数 x, y, w, h:", x, y, w, h)
    return (x, y), (w, h)

def generate_scaled_rectangles(location, size, n_scales = 4):
    """
    生成n次同心缩放后的矩形轮廓。
    :param location: 左上角坐标
    :param size: 宽高
    :param n_scales: 缩放次数
    :param scale_factor: 缩放因子
    :return: 缩放后的矩形列表
    """
    scaled_shapes = []
    x, y = location
    w, h = size
    print("原始矩形大小:", w, h)
    
    for i in range(n_scales):
        scaled_w = int(w * (1-i/n_scales))
        scaled_h = int(h * (1-i/n_scales))
        # 计算新的左上角位置,保持居中
        new_x = x + (w - scaled_w) // 2
        new_y = y + (h - scaled_h) // 2
        scaled_shapes.append(((new_x, new_y), (scaled_w, scaled_h)))
    
    return scaled_shapes

def generate_grid(location, size, cell_size, sticker_size):
    """
    在矩形中生成规则网格。
    :param location: 左上角坐标
    :param size: 矩形宽高
    :param cell_size: 网格单元大小
    :return: 网格点位置列表
    """
    x, y = location
    w, h = size
    
    rows = h // cell_size
    cols = w // cell_size
    # 新增，需测试
    row_height = (h - sticker_size)//(rows-1)
    col_width = (w - sticker_size)//(cols-1)
    grid_positions = [(x + j * col_width, y + i * row_height) 
                     for i in range(rows) for j in range(cols)]
    # 新增，需测试

    '''
    grid_positions = [(x + j * cell_size, y + i * cell_size) 
                     for i in range(rows) for j in range(cols)]
    '''
    print('网格数量:', len(grid_positions))
    return grid_positions

def transform_sticker(sticker, sticker_size, angle_open=True):
    """
    对贴纸进行缩放和旋转变换。
    :param sticker: 贴纸图片
    :param sticker_size: 贴纸大小
    :return: 变换后的贴纸
    """
    # 缩放贴纸
    long_side = max(sticker.size)
    scale_factor = (sticker_size / long_side)* random.uniform(0.95, 1.05)
    new_size = tuple(int(dim * scale_factor) for dim in sticker.size)
    sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
    
    # 随机旋转贴纸
    if angle_open:
        angle = random.uniform(-10, 10)
        sticker = sticker.rotate(angle, expand=True)
    return sticker

def place_stickers(scaled_shapes, stickers, main_sticker, transparent_bg, 
                  sticker_size, main_sticker_size, cell_size_factor):
    """
    在矩形内放置贴纸。
    :param scaled_shapes: 缩放后的矩形列表
    :param stickers: 贴纸列表
    :param main_sticker: 主贴纸
    :param transparent_bg: 透明背景
    :param sticker_size: 贴纸大小
    :param main_sticker_size: 主贴纸大小
    :param location: 矩形左上角坐标
    :param size: 矩形宽高
    :return: 贴纸拼贴结果
    """
    '''
    right_boundary = location[0] + size[0]
    bottom_boundary = location[1] + size[1]
    cell_size = int(sticker_size * cell_size_factor)
    '''
    cell_size = int(sticker_size * cell_size_factor)
    # 贴纸计数器
    count = random.randint(0, len(stickers)-1)
    # 逐层放置贴纸
    for i, shape in enumerate(scaled_shapes):
        # 获取该层的边界值
        left_boundary = shape[0][0]
        top_boundary = shape[0][1]
        right_boundary = left_boundary + shape[1][0]
        bottom_boundary = top_boundary + shape[1][1]
        # 生成网格
        grid_positions = generate_grid(shape[0], shape[1], cell_size, sticker_size)
        # 在网格点放置贴纸
        for j, pos in enumerate(grid_positions):
            sticker = stickers[count % len(stickers)]
            # transformed = transform_sticker(sticker, sticker_size)
            # 贴纸左右偏移
            x_offset = random.randint(-5, 5)
            # 处理左侧边界对齐
            if pos[0] == left_boundary:
                angle_open = False
                transformed = transform_sticker(sticker, sticker_size, angle_open)
                if bottom_boundary - pos[1] < int(2*cell_size):
                    pos = (pos[0], bottom_boundary - transformed.size[1])
            # 处理右侧边界对齐
            elif right_boundary - pos[0] < int(2*cell_size):
                angle_open = False
                transformed = transform_sticker(sticker, sticker_size, angle_open)
                pos = (right_boundary - transformed.size[0], pos[1])
                if bottom_boundary - pos[1] < int(2*cell_size):
                    pos = (right_boundary - transformed.size[0], bottom_boundary - transformed.size[1])
            # 处理顶部边界对齐
            elif pos[1] == top_boundary:
                angle_open = False
                transformed = transform_sticker(sticker, sticker_size, angle_open)
            # 处理底部边界对齐
            elif bottom_boundary - pos[1] < int(2*cell_size):
                angle_open = False
                transformed = transform_sticker(sticker, sticker_size, angle_open)
                pos = (pos[0], bottom_boundary - transformed.size[1])
            else:
                transformed = transform_sticker(sticker, sticker_size)
            # 贴纸放置
            transparent_bg.paste(transformed, (pos[0] + x_offset, pos[1]), transformed)
            count += 1

    # 放置顶层主贴纸
    main_sticker = transform_sticker(random.choice(main_sticker), main_sticker_size)
    w, h = transparent_bg.size
    pos_x = w//2 - main_sticker.size[0]//2
    pos_y = h//2 - main_sticker.size[1]//2
    transparent_bg.paste(main_sticker, (pos_x, pos_y), main_sticker)

    return transparent_bg

def rectangle_contour_main(image_path, stickers_path, main_sticker_path, 
         n_scales=4, sticker_size=220, main_sticker_size=380, cell_size_factor=0.45, crop_image=False, padding=0):
    """
    主函数。
    :param image_path: 矩形图片路径
    :param stickers_path: 贴纸文件夹路径
    :param main_sticker_path: 主贴纸路径
    :param n_scales: 缩放次数
    :param sticker_size: 贴纸大小
    :param main_sticker_size: 主贴纸大小
    :param crop_image: 是否裁剪图片
    :param padding: 裁剪时额外添加的边距    
    """
    # 加载图片和贴纸
    # shape_image, stickers = load_images(image_path, stickers_path)
    # _, main_sticker = load_images(image_path, main_sticker_path)
    shape_image = images_process(image_path)
    shape_image = shape_image[0]
    stickers = images_process(stickers_path, crop_image=crop_image, padding=padding)
    main_sticker = images_process(main_sticker_path, crop_image=crop_image, padding=padding)
    # 检测矩形并生成缩放序列
    location, size = detect_rectangle(shape_image)
    scaled_shapes = generate_scaled_rectangles(location, size, n_scales)
    # 创建透明背景并放置贴纸
    transparent_bg = Image.new("RGBA", shape_image.size, (0, 0, 0, 0))
    result = place_stickers(scaled_shapes, stickers, main_sticker,
                          transparent_bg, sticker_size, main_sticker_size, cell_size_factor)
    
    return result

if __name__ == "__main__":
    # 开始时间
    start_time = time.time()
    image_path = "images/rectangle.png"
    stickers_path = "images/baby_bear/all"
    main_sticker_path = "images/baby_bear/top"
    output_path = "images/output/output103.png"
    
    result_image = rectangle_contour_main(image_path, stickers_path, main_sticker_path,
         n_scales=4, sticker_size=280, main_sticker_size=450, cell_size_factor=0.45)
    result_image.save(output_path, format='PNG')
    print("处理完成")
    # 结束时间
    end_time = time.time()
    print(f"总时间：{end_time - start_time:.2f}秒")
