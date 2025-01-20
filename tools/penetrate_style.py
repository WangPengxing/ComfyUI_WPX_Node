import cv2
import numpy as np
from PIL import Image
import random
import time
from .image_process import images_process


def create_background(bg_width=1600, bg_height=1600):
    '''创建白色背景图'''
    background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255
    background = Image.fromarray(background)
    return background

def create_grid(background, rows=3, cols=1):
    '''创建网格区域'''
    h, w= background.size
    cell_width = w // cols
    cell_height = h // rows
    grid_cells = []
    for i in range(rows):
        for j in range(cols):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = (j + 1) * cell_width
            y2 = (i + 1) * cell_height
            grid_cells.append((x1, y1, x2, y2))
    return grid_cells

def place_sticker_set(background, sticker_sets, cell, sticker_set_cols = 3, y_cell_offset1=0, count_sticker_set=0):
    '''style1, 错落叠放合集贴纸'''
    x1, y1, x2, y2 = cell
    # 边距
    margins = (int((x2-x1)*0.1)//2, int((y2-y1)*0.1)//2)
    cell_cell_size = ((x2-x1-margins[0]*2)//sticker_set_cols, y2-y1-margins[1]*2)
    safe_frame_size = (int(cell_cell_size[0]*0.8), int(cell_cell_size[1]*0.85))
    
    # 叠加贴纸的偏移
    x_offset, y_offset = (cell_cell_size[0]-safe_frame_size[0])//3, int(safe_frame_size[1]*0.20)//3
    
    for i in range(sticker_set_cols):
        cell_cell_center = (x1 + margins[0] + i * cell_cell_size[0] + cell_cell_size[0] // 2, 
                          y1 + margins[1] + cell_cell_size[1] // 2)
        
        # 选择合集贴纸
        sticker = sticker_sets[count_sticker_set%len(sticker_sets)]
        
        # 根据安全框缩放合集贴纸
        scale_factor = (int(safe_frame_size[1]*0.87))/sticker.size[1]
        new_size = (int(sticker.size[0] * scale_factor), int(sticker.size[1] * scale_factor))
        sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
        
        # 底层合集贴纸的左上坐标
        x_base, y_base = cell_cell_center[0]+safe_frame_size[0]//2-new_size[0], cell_cell_center[1]-safe_frame_size[1]//2
        
        for j in range(4):
            x, y = x_base-j*x_offset, y_base+j*y_offset+y_cell_offset1
            if x<0:
                print("超出边界")
                continue
            # 再粘贴贴纸
            background.paste(sticker, (x, y), sticker)
            
        count_sticker_set += 1

    return background, count_sticker_set

def place_random_stickers(background, single_stickers, cell, random_stickers_rows, y_cell_offset2=0, count_single_sticker=0):
    """style2: 将贴纸随机放置到白色背景上，按行分布"""
    x1, y1, x2, y2 = cell
    print(f"style2的cell坐标{cell}")
    # 边距
    margins = (int((x2-x1)*0.1)//2, int((y2-y1)*0.1)//2)
    # 设定每行最多容纳的贴纸数量
    max_stickers_per_row = 9
    # 计算每行的高度
    row_height = (y2-y1) // random_stickers_rows
    sticker_positions = []
    sticker_resized = []
    # 按行分布贴纸
    for row in range(random_stickers_rows):
        row_top = row * row_height + y1
        row_bottom = row_top + row_height
        max_width_for_row = x2 - x1 - margins[0]  # 每行的最大宽度（含左侧边距）
        current_x = margins[0]
        
        for i in range(max_stickers_per_row):
            sticker = single_stickers[count_single_sticker%len(single_stickers)]
            # 随机选择贴纸的大小
            sticker_width, sticker_height = sticker.size
            random_scale = random.uniform(0.85, 0.95)*row_height/sticker_height  # 缩放因子
            new_width = int(sticker_width * random_scale)
            new_height = int(sticker_height * random_scale)
            if 0.5*new_width < x2 - margins[0] - current_x < new_width:
                current_x = x2 - margins[0] - new_width
            # 确保贴纸不会超出背景边界
            if current_x + new_width > max_width_for_row:
                break  # 超过最大宽度则跳出循环
                        # 重新调整贴纸大小
            transformed_sticker = sticker.resize((new_width, new_height), Image.Resampling.LANCZOS)
            sticker_resized.append(transformed_sticker)
            # 计算贴纸的位置
            position = (current_x, random.randint(row_top, row_bottom - new_height)+y_cell_offset2)
            sticker_positions.append(position)
            current_x = current_x + transformed_sticker.size[0] + random.randint(20, 30)  # 随机间隔
            count_single_sticker += 1  # 计数器

    # 将贴纸放置到背景图上
    for i, position in enumerate(sticker_positions):
        background.paste(sticker_resized[i], position, sticker_resized[i])
    return background, count_single_sticker

def place_single_sticker(background, single_stickers, cell, sticker_contour_num=3, y_cell_offset3=0, count_single_sticker=0):
    '''style3: 放置单张贴纸并绘制外接圆'''
    x1, y1, x2, y2 = cell
    print(f"style3的cell坐标{cell}")
    # 边距
    margins = (int((x2-x1)*0.1)//2, int((y2-y1)*0.1)//2)
    cell_cell_size = ((x2-x1-2*margins[0])//sticker_contour_num, y2-y1)

    for i in range(sticker_contour_num):
        cell_cell_center = (margins[0] + i * cell_cell_size[0] + cell_cell_size[0] // 2, y1 + cell_cell_size[1] // 2+y_cell_offset3)
        # 选择单张贴纸
        sticker = single_stickers[count_single_sticker%len(single_stickers)]
        # 缩放贴纸
        scale_factor = (y2 - y1)*0.7 / sticker.size[1]
        new_size = (int(sticker.size[0] * scale_factor), int(sticker.size[1] * scale_factor))
        # 外接圆��径
        radius = int(max(new_size[0], new_size[1])*1.15 / 2)
        # 限制外接圆半径
        if radius*2 > cell_cell_size[0]:
            radius = int(cell_cell_size[0]*0.9/2)
            # print("外接圆半径过大，已调整")
            if new_size[0] > new_size[1]:
                new_size = (int(radius*2*0.9), int(new_size[1]*radius*2*0.9/new_size[0]))
                # print("new_size[0] > new_size[1]")
            else:
                new_size = (int(new_size[0]*radius*2*0.9/new_size[1]), int(radius*2*0.9))
                # print("new_size[0] <= new_size[1]")
        # 贴纸放置坐标
        x_base, y_base = cell_cell_center[0]-new_size[0]//2, cell_cell_center[1]-new_size[1]//2
        # 重新调整贴纸大小
        sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
        # print(f"外接圆半径{radius}")
        # 放置贴纸并绘制外接圆
        background.paste(sticker, (x_base, y_base), sticker)
        background = np.array(background)
        cv2.circle(background, cell_cell_center, radius, (0, 0, 0), 1)
        background = Image.fromarray(background)     
        count_single_sticker += 1  # 计数器
    return background, count_single_sticker

def place_sticker_set_rotation(background, sticker_sets, cell, sticker_set_cols=4, y_cell_offset4=0, count_sticker_set=0):
    '''style4: 合集贴纸旋转放置，每列4张贴纸，带叠加和旋转效果，以左下角为锚点'''
    x1, y1, x2, y2 = cell
    print(f"style4的cell坐标{cell}")
    # 边距
    margins = (int((x2-x1)*0.1)//2, int((y2-y1)*0.1)//2)
    # 计算每列的宽度
    cell_width = (x2-x1-margins[0]*2)//sticker_set_cols
    # 计算实际贴纸宽度（列宽的80%）
    sticker_width = int(cell_width * 0.8)
    # 计算总可用宽度
    total_width = x2 - x1 - margins[0] * 2
    # 计算列间距
    spacing = (total_width - sticker_width * sticker_set_cols) // (sticker_set_cols - 1)
    
    # 贴纸的旋转角度
    rotation_angles = [10.5, 7.0, 3.5, 0]
    
    # 遍历每一列
    for col in range(sticker_set_cols):
        # 计算当前列的基准x坐标
        base_x = x1 + margins[0] + col * (sticker_width + spacing)
        # 计算基准y坐标
        base_y = y2 - margins[1]+y_cell_offset4
        # 选择贴纸
        sticker = sticker_sets[count_sticker_set%len(sticker_sets)]
        # 计算缩放因子
        scale_factor = sticker_width / sticker.size[0]
        new_size = (int(sticker.size[0] * scale_factor), int(sticker.size[1] * scale_factor))
        # 调整贴纸大小
        resized_sticker = sticker.resize(new_size, Image.Resampling.LANCZOS)
        
        # 在每列中放置4张贴纸（从下到上叠加）
        for i in range(4):
            # 沿左下角旋转贴纸
            resized_sticker_copy = resized_sticker.copy()
            rotated_sticker = resized_sticker_copy.rotate(rotation_angles[i], resample=Image.Resampling.BICUBIC, expand=True)
            # 旋转后的贴纸左上角坐标
            rotated_sticker_x = base_x-(new_size[0]*3//2)*np.sin(rotation_angles[i]*np.pi/180)
            rotated_sticker_y = y1+margins[1]-(new_size[1]//2)*np.sin(rotation_angles[i]*np.pi/180)
            # 粘贴旋转后的贴纸
            background.paste(rotated_sticker, (int(rotated_sticker_x), int(rotated_sticker_y)+margins[1]+y_cell_offset4), rotated_sticker)
        count_sticker_set += 1  # 计数器
    return background, count_sticker_set

def place_sticker_set_and_main_sticker(background, sticker_sets, main_sticker, cell, sticker_set_cols = 3, y_cell_offset5=0):
    '''style5, 错落叠放合集贴纸，中间放主贴纸'''
    x1, y1, x2, y2 = cell
    # 边距
    margins = (int((x2-x1)*0.1)//2, int((y2-y1)*0.1)//2)
    print(f"style5的cell坐标{cell}")
    cell_cell_size = ((x2-x1-margins[0]*2)//sticker_set_cols, y2-y1-margins[1]*2)
    safe_frame_size = (int(cell_cell_size[0]*0.8), int(cell_cell_size[1]*0.85))
    print(f"安全框尺寸{safe_frame_size}")
    # 叠加贴纸的偏移
    x_offset, y_offset = (cell_cell_size[0]-safe_frame_size[0])//3, int(safe_frame_size[1]*0.20)//3
    print(f"style5的x_offset, y_offset{x_offset, y_offset}")
    # 随机因子
    random_index = random.randint(1, len(sticker_sets))
    # 随机放置合集贴纸的大小（缩放合集贴纸）
    for i in range(sticker_set_cols):
        if i == 1:
            scale_factor = (y2-y1)/main_sticker.size[1]
            new_size_main = (int(main_sticker.size[0] * scale_factor), int(main_sticker.size[1] * scale_factor))
            main_sticker = main_sticker.resize(new_size_main, Image.BICUBIC)
            position = (x2//2-new_size_main[0]//2, y1+y_cell_offset5)
            background.paste(main_sticker, position, main_sticker)
            continue
        cell_cell_center = (x1 + margins[0] + i * cell_cell_size[0] + cell_cell_size[0] // 2, y1 + margins[1] + cell_cell_size[1] // 2)
        # 选择合集贴纸
        sticker = sticker_sets[(i + random_index)%len(sticker_sets)]
        # 根据安全框缩放合集贴纸
        scale_factor = (int(safe_frame_size[1]*0.87))/sticker.size[1]
        new_size = (int(sticker.size[0] * scale_factor), int(sticker.size[1] * scale_factor))
        sticker = sticker.resize(new_size, Image.BICUBIC)
        # 底层合集贴纸的左上坐标
        if i == 0:
            x_base, y_base = cell_cell_center[0]+safe_frame_size[0]//2-new_size[0]-margins[0]*2//3, cell_cell_center[1]-safe_frame_size[1]//2-margins[1]  #第一列左移margins[0]*2//3
        else:
            x_base, y_base = cell_cell_center[0]+safe_frame_size[0]//2-new_size[0]+margins[0]*2//3, cell_cell_center[1]-safe_frame_size[1]//2-margins[1]    #其他列右移margins[0]*2//3
        for j in range(4):
            x, y = x_base-j*x_offset, y_base+j*y_offset+y_cell_offset5
            if x<0:
                print("超出边界")
            background.paste(sticker, (x, y), sticker)
    return background

def penetrate_style_main(sticker_set_path, single_sticker_path, circumcircle_sticker_path, main_sticker_path, bg_width, bg_height, rows, cols, styles=('style1', 'style2', 'style3'), 
        y_cell_offset=[0, 0, 0], sticker_set_cols=3, random_stickers_rows=3, sticker_contour_num=3):
    # 加载贴纸图
    # sticker_sets, single_stickers = load_images(sticker_set_path, single_sticker_path)
    sticker_sets = images_process(sticker_set_path)
    single_stickers = images_process(single_sticker_path)
    circumcircle_stickers = images_process(circumcircle_sticker_path)
    main_sticker = images_process(main_sticker_path)[0]
    print(f"贴纸个数{len(single_stickers)}")
    print(f"合集贴纸个数{len(sticker_sets)}")
    print(f"外接圆贴纸个数{len(circumcircle_stickers)}")
    # 创建白色背景和网格
    background = create_background(bg_width, bg_height)
    grid_cells = create_grid(background, rows, cols)
    # 计数器
    count_sticker_set = 0
    count_single_sticker = 0
    count_circumcircle_sticker = 0
    # 放置贴纸：按行选择不同方式
    for i, cell in enumerate(grid_cells):
        if styles[i] == 'style1':
            y_cell_offset_i = y_cell_offset[i]
            background, count_sticker_set = place_sticker_set(background, sticker_sets, cell, sticker_set_cols, y_cell_offset_i, count_sticker_set)  # 使用合集贴纸
        elif styles[i] == 'style2':
            y_cell_offset_i = y_cell_offset[i]
            background, count_single_sticker = place_random_stickers(background, single_stickers, cell, random_stickers_rows, y_cell_offset_i, count_single_sticker)  # 随机分布贴纸
        elif styles[i] == 'style3':
            y_cell_offset_i = y_cell_offset[i]
            background, count_circumcircle_sticker = place_single_sticker(background, circumcircle_stickers, cell, sticker_contour_num, y_cell_offset_i, count_circumcircle_sticker)  # 单张贴纸展示
        elif styles[i] == 'style4':
            y_cell_offset_i = y_cell_offset[i]
            background, count_sticker_set = place_sticker_set_rotation(background, sticker_sets, cell, sticker_set_cols, y_cell_offset_i, count_sticker_set)  # 合集贴纸旋转放置
        elif styles[i] == 'style5':
            y_cell_offset_i = y_cell_offset[i]
            background = place_sticker_set_and_main_sticker(background, sticker_sets, main_sticker, cell, sticker_set_cols, y_cell_offset_i)  # 合集贴纸旋转放置，中间放主贴纸
    # 保存结果
    result_img = background
    return result_img

if __name__ == "__main__":
    # 开始时间 
    start_time = time.time()
    sticker_set_path = "images/sticker_set"   # 贴纸合集路径
    single_sticker_path = "images/baby_bear/all"  # 单张贴纸路径
    circumcircle_sticker_path = "images/baby_bear/all"  # 外接圆贴纸路径
    main_sticker_path = "images/stickers_folder/散装_画板 1 副本 97.png"  # 主贴纸路径
    output_path = "images/output/output112.png"   # 输出文件路径
    cell1_offset = 0
    cell2_offset = 0
    cell3_offset = 0
    #'''
    result_img = penetrate_style_main(sticker_set_path, single_sticker_path, circumcircle_sticker_path, main_sticker_path, bg_width=1600, bg_height=1600, rows=3, cols=1, styles=('style1', 'style2', 'style2'), 
        y_cell_offset=(cell1_offset, cell2_offset, cell3_offset), sticker_set_cols=3, random_stickers_rows=3, sticker_contour_num = 4)
    result_img.save(output_path, format='PNG')
    #'''
    # 结束时间
    end_time = time.time()
    print(f"总时间：{end_time - start_time:.2f}秒")
    s = Image.open("images/sticker_set/资源 105.png")
    s = create_smooth_shadow(s, blur_radius=4, offset=(100, 100), shadow_color=(0,0,0,0))
    s.save("images/output/output113.png", format='PNG')

