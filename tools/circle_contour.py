import time 
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from .animal_contour import animal_contour_main
from .image_process import images_process
import os
import random

def detect_base_shape(shape_image):
    """检测基础形状，并返回类型、中心点坐标或左上坐标、半径或宽高"""
    # 转换为灰度图并二值化
    gray_img = np.array(shape_image.convert("L"))
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)  # 最大轮廓
    x, y, w, h = cv2.boundingRect(cnt)
    # 圆心和半径
    radius = w // 2
    center = (x + radius, y + radius)
    return center, radius


def generate_scaled_shapes(center, radius, n_scales):
    """生成n次同心缩放后的形状轮廓"""
    scale_shapes = []
    for i in range(n_scales):
        scaled_radius = int(radius*(1-i/(n_scales+1))) 
        scale_shapes.append((center, scaled_radius))
    return scale_shapes

def generate_circle_contour(shape_image, n_scales):
    """生成圆形轮廓"""
    center, radius = detect_base_shape(shape_image)
    scale_shapes = generate_scaled_shapes(center, radius, n_scales)
    circle_contours = []
    # 创建黑色背景，将圆形轮廓绘制在背景上，内填充白色 
    for center, radius in scale_shapes:
        black_bg = Image.new("RGBA", shape_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(black_bg) # 创建绘制对象
        # 绘制圆形轮廓，内填充白色
        draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), outline="white", fill="white")
        circle_contours.append(black_bg)
    return circle_contours

def circle_contour_main(image_path, stickers_paths, main_sticker_path, n_scales, 
                        max_sticker_count, spacing, overlap_threshold_contour, bg_sticker_size, main_sticker_size,
                        crop_image=False, padding=0):
    """主函数，处理图像并放置贴纸"""
    # shape_image = load_images(image_path)
    shape_image = images_process(image_path)
    shape_image = shape_image[0]
    circle_contours = generate_circle_contour(shape_image, n_scales)
    transparent_bg = Image.new("RGBA", shape_image.size, (0, 0, 0, 0))
    sticker_index = random.randint(0, 49)
    for i, circle_contour in enumerate(circle_contours):
        max_sticker_count = max_sticker_count // (i+1)
        print(f"第{i+1}层最大贴纸数目：{max_sticker_count}")
        result_image, sticker_index = animal_contour_main(
            image_path = circle_contour,
            stickers_paths = stickers_paths,
            main_sticker_path = main_sticker_path,
            max_sticker_count = max_sticker_count,
            spacing = spacing,
            overlap_threshold_contour = overlap_threshold_contour,
            bg_sticker_size = bg_sticker_size,
            main_sticker_size = main_sticker_size,
            sticker_index = sticker_index,
            crop_image = crop_image,
            padding = padding
        )
        transparent_bg = Image.alpha_composite(transparent_bg, result_image)
    return transparent_bg

if __name__ == "__main__":
    # 开始时间
    time_start = time.time()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = root.replace("\\", "/") + "/"
    result_image = circle_contour_main(
        image_path = os.path.join(root, "images/circle2.png"),
        stickers_paths = os.path.join(root, "images/baby_bear/all"),
        main_sticker_path = os.path.join(root, "images/baby_bear/all/图层 2.png"),
        n_scales = 3,
        max_sticker_count = 80,
        spacing = 30,
        overlap_threshold_contour = 0.6,
        bg_sticker_size = 330,
        main_sticker_size = 450
    )
    result_image.save(os.path.join(root, "images/output/output101.png"), format='PNG')
    # 结束时间
    time_end = time.time()
    print(f"总时间：{time_end - time_start:.2f}秒")

