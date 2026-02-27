#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================
#  导入所需的 Python 库
# ===========================

import cv2  # 导入 OpenCV 库，用来读取和处理图片
import mediapipe as mp  # 导入 Mediapipe 库，用来做人脸关键点检测
import numpy as np  # 导入 Numpy 数学库，用来做向量和距离计算
import math  # 导入数学库，用来做一些数学运算（例如开方等）
import random  # 导入随机数库，用来生成“彩蛋”随机文案
from PIL import Image, ImageDraw, ImageFont  # 导入 Pillow 库，用来绘制支持中文的文字


# ===========================
#  工具函数：在 OpenCV 图像上绘制中文文本
# ===========================
def cv2_add_chinese_text(cv2_img, text, position, font_size=24, color=(255, 255, 255)):
    """
    在 OpenCV 的 BGR 图像上绘制支持中文的文字。

    实现思路：
    1. 把 OpenCV 图像从 BGR 格式转换为 RGB，并转成 PIL Image。
    2. 使用 Pillow 的 ImageDraw + ImageFont 来绘制中文。
    3. 再把 PIL 图像转换回 OpenCV 使用的 BGR numpy 数组。

    参数:
        cv2_img: 原始 OpenCV 图像（BGR 格式的 numpy 数组）
        text: 要绘制的中文字符串
        position: 文本左上角坐标 (x, y)
        font_size: 字体大小
        color: 字体颜色（BGR 格式，例如 (255, 255, 255) 为白色）

    返回:
        绘制好文字之后的 OpenCV 图像（仍然是 BGR 格式）
    """

    # 1. 把 BGR 转成 RGB，然后用 Image.fromarray 转成 PIL 图像
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)

    # 2. 准备绘图对象
    draw = ImageDraw.Draw(pil_img)

    # 3. 依次尝试加载几种常见的系统中文字体，只要有一个加载成功就可以正常显示中文
    #    不同 macOS 版本可能字体路径略有差异，这里做一个“候选列表”
    candidate_fonts = [
        "/System/Library/Fonts/PingFang.ttc",           # 苹方（新系统常用）
        "/System/Library/Fonts/STHeiti Medium.ttc",     # 华文黑体
        "/System/Library/Fonts/STHeiti Light.ttc",      # 华文细黑
        "/System/Library/Fonts/STSong.ttf",             # 华文宋体
    ]

    font = None
    last_error = None
    for path in candidate_fonts:
        try:
            font = ImageFont.truetype(path, font_size)
            # 一旦加载成功就停止尝试
            # print(f"使用字体: {path}")  # 如需调试可打开
            break
        except Exception as e:
            last_error = e
            continue

    # 如果所有指定字体都加载失败，则退回到默认字体（可能不完全支持中文）
    if font is None:
        print(f"警告：无法加载系统中文字体，已退回 Pillow 默认字体。最后一次错误信息：{last_error}")
        font = ImageFont.load_default()

    # 4. 由于 Pillow 使用的是 RGB 颜色，这里需要把 BGR 转成 RGB
    b, g, r = color
    color_rgb = (r, g, b)

    # 5. 绘制文本（Pillow 坐标系与 OpenCV 一致，左上角为原点）
    draw.text(position, text, font=font, fill=color_rgb)

    # 6. 把 PIL 图像转换回 OpenCV 的 BGR numpy 数组
    cv2_img_with_text = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img_with_text

# ===========================
#  初始化 Mediapipe 的人脸网格 (Face Mesh) 模型
# ===========================

mp_face_mesh = mp.solutions.face_mesh  # 从 mediapipe 中拿到人脸网格的模块
mp_draw = mp.solutions.drawing_utils  # 可选：用来画关键点和连接线（本例主要用数据，不画图）


# ===========================
#  工具函数：计算两点间的欧式距离
# ===========================
def euclidean_distance(p1, p2):
    """
    计算两点之间的直线距离

    参数:
        p1: 第一个点的 (x, y) 坐标
        p2: 第二个点的 (x, y) 坐标

    返回:
        两点之间的欧式距离（一个浮点数）
    """
    # np.linalg.norm 是 Numpy 提供的“向量长度”函数，这里用它算两点差的长度
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# ===========================
#  工具函数：从 468 个关键点中，提取我们关心的几个点
# ===========================
def extract_key_points(landmarks, image_width, image_height):
    """
    从 Mediapipe 返回的人脸关键点中，提取“三庭五眼”和对称度需要用到的关键点。

    参数:
        landmarks: Mediapipe Face Mesh 的关键点列表（每个点是归一化坐标）
        image_width: 图片的宽度（像素）
        image_height: 图片的高度（像素）

    返回:
        一个字典，里面是我们需要的关键点的像素坐标 (x, y)
    """

    # 下面这些 index 是 Mediapipe Face Mesh 官方定义的人脸各个部位的索引，
    # 通过这些索引，我们可以大致拿到额头、眉毛、眼睛、鼻子、下巴、脸颊边缘等位置的点。
    #
    # 注意：Face Mesh 总共有 468 个点，这里只挑一些典型位置来近似表示“三庭五眼”。

    key_points_index = {
        # 额头大致中心（这里用靠近眉毛上方的点近似）
        "forehead": 10,  # 10 号点大致在额头偏上的位置

        # 左右眉毛上缘中点（近似表示眉线高度）
        "left_eyebrow": 105,   # 左眉中上方
        "right_eyebrow": 334,  # 右眉中上方

        # 左右眼睛外眼角、内眼角
        "left_eye_outer": 33,   # 左眼外眼角
        "left_eye_inner": 133,  # 左眼内眼角

        "right_eye_inner": 362,  # 右眼内眼角
        "right_eye_outer": 263,  # 右眼外眼角

        # 鼻尖
        "nose_tip": 1,  # 1 号点大致是鼻尖

        # 下巴最底部
        "chin": 152,  # 下巴最低点

        # 左右脸颊最外侧（脸型轮廓两侧）
        "left_face_outer": 234,   # 左侧脸部轮廓外沿
        "right_face_outer": 454,  # 右侧脸部轮廓外沿

        # 鼻梁中点附近（用来估计左右对称的中心轴）
        "nose_bridge": 168,  # 鼻梁中间偏上的点
    }

    # 创建一个字典用来保存转换成像素坐标后的关键点
    points = {}

    # 遍历我们定义的关键点索引
    for name, idx in key_points_index.items():
        # 从 landmarks 中取出对应序号的点
        lm = landmarks[idx]

        # Mediapipe 给的是归一化坐标（0~1 之间），这里乘以图片宽高得到像素坐标
        x = lm.x * image_width
        y = lm.y * image_height

        # 保存到 points 字典中，格式为 (x, y)
        points[name] = (x, y)

    return points


# ===========================
#  计算三庭比例
# ===========================
def compute_three_courts(points):
    """
    计算“三庭”比例：
    - 上庭：额头到眉眼的高度
    - 中庭：眉眼到鼻尖的高度
    - 下庭：鼻尖到下巴的高度

    参数:
        points: extract_key_points 返回的关键点像素坐标字典

    返回:
        一个字典，包含每一庭的绝对高度、总高度、以及相对比例
    """

    # 为了简单，我们用以下点来近似三庭分界：
    # - 额头点 forehead
    # - 眉眼中线：左右眉毛点的中点
    # - 鼻尖 nose_tip
    # - 下巴 chin

    forehead = points["forehead"]
    left_eyebrow = points["left_eyebrow"]
    right_eyebrow = points["right_eyebrow"]
    nose_tip = points["nose_tip"]
    chin = points["chin"]

    # 计算眉眼中线点：左右眉毛点的中点
    eyebrow_center = (
        (left_eyebrow[0] + right_eyebrow[0]) / 2.0,
        (left_eyebrow[1] + right_eyebrow[1]) / 2.0,
    )

    # 三庭的高度：用 y 方向的绝对差来近似（竖直方向）
    # 注意：图像坐标中 y 向下为正，所以这里用绝对值保证为正数
    upper = abs(forehead[1] - eyebrow_center[1])   # 上庭：额头到眉眼
    middle = abs(eyebrow_center[1] - nose_tip[1])  # 中庭：眉眼到鼻尖
    lower = abs(nose_tip[1] - chin[1])             # 下庭：鼻尖到下巴

    # 计算总高度（三庭相加）
    total = upper + middle + lower if (upper + middle + lower) > 0 else 1.0

    # 计算每一庭占总高度的比例
    upper_ratio = upper / total
    middle_ratio = middle / total
    lower_ratio = lower / total

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "total": total,
        "upper_ratio": upper_ratio,
        "middle_ratio": middle_ratio,
        "lower_ratio": lower_ratio,
    }


# ===========================
#  计算五眼比例
# ===========================
def compute_five_eyes(points):
    """
    计算“五眼”相关的水平距离：
    - 左脸轮廓到左眼外眼角的距离
    - 左眼宽度（外眼角到内眼角）
    - 眼间距（两眼内眼角之间的距离）
    - 右眼宽度（内眼角到外眼角）
    - 右眼外眼角到右脸轮廓的距离

    参数:
        points: extract_key_points 返回的关键点像素坐标字典

    返回:
        一个字典，包含这些绝对宽度和相对比例
    """

    # 从 points 取出我们需要使用的点
    left_face_outer = points["left_face_outer"]
    right_face_outer = points["right_face_outer"]

    left_eye_outer = points["left_eye_outer"]
    left_eye_inner = points["left_eye_inner"]

    right_eye_inner = points["right_eye_inner"]
    right_eye_outer = points["right_eye_outer"]

    # 计算各段水平距离（只考虑 x 方向的差）
    # 为简单起见，这里直接用 x 坐标差的绝对值来代表水平距离
    left_margin = abs(left_eye_outer[0] - left_face_outer[0])  # 左脸到左眼外眼角
    left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])  # 左眼宽度

    eye_gap = abs(left_eye_inner[0] - right_eye_inner[0])  # 两眼内眼角之间的距离（眼间距）

    right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])  # 右眼宽度
    right_margin = abs(right_face_outer[0] - right_eye_outer[0])  # 右眼外眼角到右脸

    # 五段相加得到总宽度
    total = left_margin + left_eye_width + eye_gap + right_eye_width + right_margin
    if total == 0:
        total = 1.0  # 防止除零

    # 计算每一段占总宽度的比例
    left_margin_ratio = left_margin / total
    left_eye_ratio = left_eye_width / total
    eye_gap_ratio = eye_gap / total
    right_eye_ratio = right_eye_width / total
    right_margin_ratio = right_margin / total

    return {
        "left_margin": left_margin,
        "left_eye_width": left_eye_width,
        "eye_gap": eye_gap,
        "right_eye_width": right_eye_width,
        "right_margin": right_margin,
        "total": total,
        "left_margin_ratio": left_margin_ratio,
        "left_eye_ratio": left_eye_ratio,
        "eye_gap_ratio": eye_gap_ratio,
        "right_eye_ratio": right_eye_ratio,
        "right_margin_ratio": right_margin_ratio,
    }


# ===========================
#  计算左右对称度
# ===========================
def compute_symmetry(points):
    """
    计算人脸左右对称度。

    思路：
    - 首先用鼻梁点和下巴点，估计出一条“面部中心竖直轴”的位置（x 坐标）。
    - 然后对左右成对的点（如左脸外轮廓和右脸外轮廓、左眼外角和右眼外角等），
      分别计算它们到这条中心轴的水平距离，比较两边距离是否接近。
    - 距离越接近，说明越对称。

    参数:
        points: extract_key_points 返回的关键点像素坐标字典

    返回:
        一个字典，包含：
        - symmetry_score: 0~1 之间的对称度评分（1 表示非常对称）
        - detail: 每一对点的具体偏差情况
    """

    # 取出用来估计中心轴的关键点：鼻梁和下巴
    nose_bridge = points["nose_bridge"]
    chin = points["chin"]

    # 中心轴的 x 坐标：取鼻梁和下巴的 x 坐标平均值
    center_x = (nose_bridge[0] + chin[0]) / 2.0

    # 定义几组左右成对的点，用来评估对称：
    # 每一项是 (左侧点名字, 右侧点名字)
    pairs = [
        ("left_face_outer", "right_face_outer"),
        ("left_eye_outer", "right_eye_outer"),
        ("left_eye_inner", "right_eye_inner"),
        ("left_eyebrow", "right_eyebrow"),
    ]

    deviations = []  # 用来存每一对的“非对称程度”（差值）
    pair_details = []  # 详细记录每一对的距离信息，便于后面打印

    for left_name, right_name in pairs:
        left_point = points[left_name]
        right_point = points[right_name]

        # 左右点到中心轴的水平距离（取绝对值，保证为正）
        left_dist = abs(left_point[0] - center_x)
        right_dist = abs(right_point[0] - center_x)

        # 两侧距离差：差值越小，说明越对称
        diff = abs(left_dist - right_dist)

        deviations.append(diff)

        pair_details.append({
            "pair": f"{left_name} vs {right_name}",
            "left_distance_to_center": left_dist,
            "right_distance_to_center": right_dist,
            "diff": diff,
        })

    # 为了把“差值”转换成 0~1 的“对称度分数”，
    # 我们需要一个“参考尺度”，可以用脸的总宽度来归一化。
    face_width = abs(points["right_face_outer"][0] - points["left_face_outer"][0])
    if face_width == 0:
        face_width = 1.0  # 防止除零

    # 把每一对的差值除以脸宽，得到“相对差”，然后取平均，得到“平均非对称程度”
    normalized_deviations = [d / face_width for d in deviations]
    avg_deviation = float(np.mean(normalized_deviations))

    # 把“非对称程度”转成“对称度分数”：分数 = 1 - (非对称程度)
    # 再限定在 [0, 1] 范围内
    symmetry_score = 1.0 - avg_deviation
    symmetry_score = max(0.0, min(1.0, symmetry_score))

    return {
        "center_x": center_x,
        "symmetry_score": symmetry_score,
        "pair_details": pair_details,
    }


# ===========================
#  把几何指标转换成 0-100 分的评分模型
# ===========================
def score_face_geometry(three_courts, five_eyes, symmetry):
    """
    根据三庭、五眼、对称度三个方面，给出一个 0~100 的综合几何分。

    说明：
    - 这只是一个非常简单、主观的打分模型，仅供娱乐参考。
    - 实际审美远比这些指标复杂得多。

    参数:
        three_courts: compute_three_courts 的返回结果
        five_eyes: compute_five_eyes 的返回结果
        symmetry: compute_symmetry 的返回结果

    返回:
        一个字典，包含：
        - geometry_score: 0~100 的综合几何分
        - sub_scores: 各子项的分数和误差详情
    """

    # ===========================
    # 1. 三庭理想比例打分
    # ===========================
    # 传统上认为 “三庭等长” 比较理想，即 上庭:中庭:下庭 ≈ 1:1:1。
    # 这里我们直接用每一庭的比例与 1/3 的差距来算。
    ideal_ratio_three = 1.0 / 3.0  # 理想的每一庭比例

    upper_ratio = three_courts["upper_ratio"]
    middle_ratio = three_courts["middle_ratio"]
    lower_ratio = three_courts["lower_ratio"]

    # 计算每一庭与理想值 1/3 的偏差（绝对值）
    diff_upper = abs(upper_ratio - ideal_ratio_three)
    diff_middle = abs(middle_ratio - ideal_ratio_three)
    diff_lower = abs(lower_ratio - ideal_ratio_three)

    # 取平均偏差来整体衡量偏离程度
    avg_diff_three = (diff_upper + diff_middle + diff_lower) / 3.0

    # 把偏差映射到 0~1 的“接近程度”：越接近 1/3，分数越高
    # 经验上，若偏差在 0.1 以内已经比较接近了，这里用 0.2 作为最大合理偏差进行归一化。
    three_score = 1.0 - min(avg_diff_three / 0.2, 1.0)  # 限制到 [0,1]

    # ===========================
    # 2. 五眼理想比例打分
    # ===========================
    # 传统“五眼”观念：理想状态下，
    # - 左边脸宽 ≈ 1 眼宽
    # - 左眼宽 ≈ 1 眼宽
    # - 眼间距 ≈ 1 眼宽
    # - 右眼宽 ≈ 1 眼宽
    # - 右边脸宽 ≈ 1 眼宽
    # 也就是五段大致相等，每段 ≈ 1/5。
    ideal_ratio_five = 1.0 / 5.0  # 五等分理想比例

    # 实际每一段比例
    ratios_five = [
        five_eyes["left_margin_ratio"],
        five_eyes["left_eye_ratio"],
        five_eyes["eye_gap_ratio"],
        five_eyes["right_eye_ratio"],
        five_eyes["right_margin_ratio"],
    ]

    # 计算每一段与理想值 1/5 的偏差（绝对值），再求平均
    diffs_five = [abs(r - ideal_ratio_five) for r in ratios_five]
    avg_diff_five = float(np.mean(diffs_five))

    # 同样用 0.2 作为最大合理偏差进行归一化
    five_score = 1.0 - min(avg_diff_five / 0.2, 1.0)  # 限制到 [0,1]

    # ===========================
    # 3. 对称度打分
    # ===========================
    # compute_symmetry 已经给出了 0~1 的对称度分数
    symmetry_score = symmetry["symmetry_score"]  # 直接使用

    # ===========================
    # 4. 综合得分（简单加权平均）
    # ===========================
    # 可以自由设定权重，例如：
    # 三庭 30%，五眼 30%，对称度 40%
    w_three = 0.3
    w_five = 0.3
    w_sym = 0.4

    # 计算综合分数（0~1）
    final_score_0_1 = three_score * w_three + five_score * w_five + symmetry_score * w_sym

    # 转换为 0~100 分
    final_score_0_100 = final_score_0_1 * 100.0

    return {
        "geometry_score": final_score_0_100,
        "sub_scores": {
            "three_courts_score_0_1": three_score,
            "five_eyes_score_0_1": five_score,
            "symmetry_score_0_1": symmetry_score,
            "three_courts_avg_diff": avg_diff_three,
            "five_eyes_avg_diff": avg_diff_five,
        },
    }


# ===========================
#  根据几何特征推断“风格标签”
# ===========================
def classify_style_tag(three_courts, five_eyes, symmetry, points):
    """
    根据三庭、五眼、对称度以及脸型比例，粗略给出一个“风格标签”。

    说明：
    - 这里只是非常粗糙的规则，更多是娱乐向的性格/风格标签，不是严肃的人类学结论。
    """

    # 提取需要用到的一些比例和指标
    middle_ratio = three_courts["middle_ratio"]  # 中庭占比

    # 两只眼睛总宽度占五眼总宽度的比例，近似“眼睛存在感”
    eye_width_ratio = (
        five_eyes["left_eye_width"] + five_eyes["right_eye_width"]
    ) / max(five_eyes["total"], 1e-6)

    sym_score = symmetry["symmetry_score"]  # 对称度

    # 近似脸型比例：脸高 / 脸宽
    face_height = three_courts["total"]
    face_width = abs(points["right_face_outer"][0] - points["left_face_outer"][0])
    if face_width <= 0:
        face_width = 1.0
    face_ratio = face_height / face_width  # 数值越大，脸越“长”；越小越“圆”

    # 默认标签
    style = "初恋脸"

    # 一些很粗糙的风格判定规则（可以按需要再微调）
    if eye_width_ratio >= 0.42 and middle_ratio <= 0.33 and face_ratio <= 1.25:
        # 眼睛存在感很强，中庭偏短，脸偏圆润
        style = "猫系"
    elif eye_width_ratio <= 0.36 and middle_ratio >= 0.35 and face_ratio >= 1.35:
        # 眼睛相对内敛，中庭略长，脸偏长
        style = "盐系"
    elif sym_score >= 0.9 and 1.25 < face_ratio < 1.4:
        # 对称度非常高、比例比较“标准”
        style = "犬系"
    elif middle_ratio < 0.32 and face_ratio > 1.35:
        # 中庭略短，脸偏长，有一点攻击性美
        style = "狐系"
    else:
        # 兜底标签：根据对称度分区再细化“初恋脸”感觉
        if sym_score >= 0.92:
            style = "初恋脸"
        elif sym_score >= 0.85:
            style = "温柔系"
        else:
            style = "个性派"

    return {
        "style_tag": style,
        "face_ratio": face_ratio,
        "eye_width_ratio": eye_width_ratio,
        "middle_ratio": middle_ratio,
    }


# ===========================
#  “夸夸文案”生成引擎
# ===========================
def generate_praise_text(score, three_courts, five_eyes, symmetry, style_info):
    """
    根据综合分数、子项最高分、风格标签，生成一段走心的夸夸文案。

    参数:
        score: score_face_geometry 的返回结果
        three_courts / five_eyes / symmetry: 用于针对性描述
        style_info: classify_style_tag 的返回结果（包含 style_tag 等）

    返回:
        一个字典，包含主评价文案和可能的“彩蛋”描述
    """

    geometry_score = score["geometry_score"]
    sub = score["sub_scores"]

    # 找出哪个子项分数最高：三庭、五眼、对称度
    sub_order = [
        ("三庭比例", sub["three_courts_score_0_1"], "three"),
        ("五眼协调", sub["five_eyes_score_0_1"], "five"),
        ("左右对称", sub["symmetry_score_0_1"], "sym"),
    ]
    sub_order.sort(key=lambda x: x[1], reverse=True)
    best_name, best_value, best_key = sub_order[0]

    # 先根据总分确定“整体语气”
    if geometry_score >= 90:
        base_text = "整体五官非常高级，属于在人群中一眼就会被记住的类型。"
    elif geometry_score >= 80:
        base_text = "整体比例非常耐看，属于越看越顺眼、非常适合近距离欣赏的类型。"
    elif geometry_score >= 70:
        base_text = "整体比例十分协调，自然舒适，有一种不费力的好看感。"
    elif geometry_score >= 60:
        base_text = "整体比例比较均衡，是那种真实生活中很耐看的长相。"
    else:
        base_text = "整体比例有自己独特的节奏感，反而塑造出非常有记忆点的气质。"

    # 再根据最高分项，给一个“专业味”更强的侧重点描述
    if best_key == "sym":
        focus_text = "你的脸型拥有艺术品般的对称美，在照片和视频里都非常上镜。"
    elif best_key == "five":
        focus_text = "你的五眼比例非常舒服，眼睛的位置和间距让整张脸看起来既自然又高级。"
    else:  # "three"
        focus_text = "你的三庭比例很在线，额头、鼻尖和下巴之间的节奏感营造出非常和谐的立体感。"

    # 风格标签补充一句话，和上面的内容自然衔接
    style_tag = style_info["style_tag"]
    if style_tag == "猫系":
        style_text = "整体气质偏猫系，带着一点软萌又有点小钝感的攻击性美，非常适合清冷或文艺风格的穿搭。"
    elif style_tag == "犬系":
        style_text = "整体气质偏犬系，看起来非常亲切、有安全感，是那种第一眼就很让人放松的长相。"
    elif style_tag == "狐系":
        style_text = "整体气质偏狐系，自带一点锋利感和神秘感，非常适合精致妆容和氛围感大片。"
    elif style_tag == "盐系":
        style_text = "整体气质偏盐系，干净、克制、不张扬，属于气质越简单越能衬托出高级感的类型。"
    elif style_tag == "初恋脸":
        style_text = "整体气质就是典型的“初恋脸”，干净、柔和，让人很容易产生亲近感。"
    elif style_tag == "温柔系":
        style_text = "整体气质非常温柔耐看，属于很适合自然光和生活化场景的脸。"
    else:  # 个性派等
        style_text = "整体气质非常有个性，有着和别人明显区分开的辨识度，是属于镜头特别眷顾的那一类人。"

    full_text = base_text + focus_text + style_text

    # 10% 概率生成一个“彩蛋”——独特魅力点
    easter_egg_text = None
    if random.random() < 0.10:
        easter_candidates = [
            "识别到非常有辨识度的卧蚕，会在微笑时瞬间放大你的亲和力。",
            "唇形曲线非常好看，适合尝试更大胆一点的唇色，会有惊喜效果。",
            "下颌线条干净利落，上镜时只要稍微调整角度，就能拥有封面级侧颜。",
            "鼻梁和鼻尖的过渡非常自然，是那种怎么拍都很真实耐看的鼻型。",
            "眼尾的走势很好看，轻轻上挑的弧度会让表情显得既温柔又有力量。",
        ]
        easter_egg_text = random.choice(easter_candidates)

    return {
        "style_tag": style_tag,
        "praise_text": full_text,
        "best_item_name": best_name,
        "easter_egg": easter_egg_text,
    }


# ===========================
#  主函数：对一张人脸图片做“几何颜值评分”，并在图像上做可视化标注
# ===========================
def analyze_face(image_path, save_visualization=True, output_path="visualized_result.jpg"):
    """
    读取一张本地图片，识别人脸关键点，计算三庭、五眼、对称度，并给出几何颜值分。

    参数:
        image_path: 图片路径（例如 "test.jpg"）
        save_visualization: 是否保存带有可视化标注的结果图片（默认保存）
        output_path: 可视化结果图片的保存路径（默认 "visualized_result.jpg"）

    返回:
        一个字典，包含：
        - three_courts: 三庭计算结果
        - five_eyes: 五眼计算结果
        - symmetry: 对称度计算结果
        - score: 综合几何分及子项分数
    """

    # 用 OpenCV 读取图片（BGR 格式）
    image = cv2.imread(image_path)

    # 如果 image 是 None，说明图片路径不对或者文件不存在
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}，请确认文件是否存在且路径正确。")

    # 获取图片的高和宽
    h, w, _ = image.shape

    # Mediapipe 的 FaceMesh 需要传入 RGB 图片，所以先从 BGR 转成 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 with 语句初始化 FaceMesh 模型，这样用完会自动释放资源
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,      # static_image_mode=True 表示这是静态图片，而不是视频流
        max_num_faces=1,            # 最多检测 1 张脸
        refine_landmarks=True,      # 是否精细化关键点（眼睛/嘴唇等更多点）
        min_detection_confidence=0.5  # 检测置信度阈值（越高越严格）
    ) as face_mesh:

        # 把 RGB 图片传给 FaceMesh 做人脸检测和关键点预测
        results = face_mesh.process(image_rgb)

        # 如果没有检测到人脸（multi_face_landmarks 为 None）
        if not results.multi_face_landmarks:
            raise ValueError("图片中未检测到清晰的人脸，请更换一张正面、光线较好的照片。")

        # 这里只取第一张脸
        face_landmarks = results.multi_face_landmarks[0]

        # 从 468 个关键点中提取我们关心的若干关键点（像素坐标）
        points = extract_key_points(face_landmarks.landmark, w, h)

        # 计算三庭比例
        three_courts = compute_three_courts(points)

        # 计算五眼比例
        five_eyes = compute_five_eyes(points)

        # 计算左右对称度
        symmetry = compute_symmetry(points)

        # 根据以上几何指标打一个 0~100 的分
        score = score_face_geometry(three_courts, five_eyes, symmetry)

        # 根据几何特征给出风格标签
        style_info = classify_style_tag(three_courts, five_eyes, symmetry, points)

        # 生成夸夸文案和可能的彩蛋
        praise_info = generate_praise_text(score, three_courts, five_eyes, symmetry, style_info)

        # ===========================
        #  可视化：在原图上绘制关键点、辅助线和分数 + 情绪价值文案
        # ===========================
        if save_visualization:
            # 复制一份原图，避免直接修改原始数据
            vis_image = image.copy()

            # ---------- 1. 绘制 468 个面部关键点 ----------
            # 遍历所有关键点（landmark 是 0~1 的归一化坐标，这里要转换成像素坐标）
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                # 在关键点位置画一个小红点（BGR: 红色 = (0, 0, 255)）
                cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)

            # ---------- 2. 绘制“三庭”的水平辅助线（绿色） ----------
            # 三庭使用的关键点：额头 forehead、眉眼中线、鼻尖 nose_tip、下巴 chin
            forehead = points["forehead"]
            left_eyebrow = points["left_eyebrow"]
            right_eyebrow = points["right_eyebrow"]
            nose_tip = points["nose_tip"]
            chin = points["chin"]

            # 眉眼中线的 y 值（和三庭计算逻辑保持一致）
            eyebrow_center_y = (left_eyebrow[1] + right_eyebrow[1]) / 2.0

            # 为了让辅助线覆盖整张脸，这里用整张图片的宽度
            x_start = 0
            x_end = w - 1

            # 额头水平线
            cv2.line(
                vis_image,
                (x_start, int(forehead[1])),
                (x_end, int(forehead[1])),
                (0, 255, 0),  # 绿色
                1,
            )
            # 眉眼水平线
            cv2.line(
                vis_image,
                (x_start, int(eyebrow_center_y)),
                (x_end, int(eyebrow_center_y)),
                (0, 255, 0),
                1,
            )
            # 鼻尖水平线
            cv2.line(
                vis_image,
                (x_start, int(nose_tip[1])),
                (x_end, int(nose_tip[1])),
                (0, 255, 0),
                1,
            )
            # 下巴水平线
            cv2.line(
                vis_image,
                (x_start, int(chin[1])),
                (x_end, int(chin[1])),
                (0, 255, 0),
                1,
            )

            # ---------- 3. 绘制“五眼”的垂直辅助线（蓝色） ----------
            # 使用五眼相关的关键点：左右脸轮廓、左右眼外眼角/内眼角
            left_face_outer = points["left_face_outer"]
            right_face_outer = points["right_face_outer"]
            left_eye_outer = points["left_eye_outer"]
            left_eye_inner = points["left_eye_inner"]
            right_eye_inner = points["right_eye_inner"]
            right_eye_outer = points["right_eye_outer"]

            # 五眼分割上的 6 条竖线的 x 坐标（从左到右）
            x_positions = [
                int(left_face_outer[0]),
                int(left_eye_outer[0]),
                int(left_eye_inner[0]),
                int(right_eye_inner[0]),
                int(right_eye_outer[0]),
                int(right_face_outer[0]),
            ]

            # 垂直线的上下范围：为了简化，这里直接从整张图片的顶部到底部
            y_top = 0
            y_bottom = h - 1

            for x_pos in x_positions:
                cv2.line(
                    vis_image,
                    (x_pos, y_top),
                    (x_pos, y_bottom),
                    (255, 0, 0),  # 蓝色（BGR）
                    1,
                )

            # ---------- 4. 在左上角写上几何评分 ----------
            # 从 score 结果中拿到几何分
            score_value = score["geometry_score"]
            text = f"几何评分: {score_value:.2f} 分"

            # 使用支持中文的绘制函数在图像左上角写字
            vis_image = cv2_add_chinese_text(
                vis_image,
                text,
                (10, 30),  # 文本左上角坐标（注意：Pillow 的 text 是以左上角为基准）
                font_size=24,
                color=(255, 255, 255),  # 白色文字（BGR）
            )

            # ---------- 5. 在图像下方绘制“情绪价值”半透明气泡 ----------
            # 组合需要展示的文案：风格标签 + 核心评价 + 彩蛋（如果有）
            bubble_lines = []
            bubble_lines.append(f"风格标签：{praise_info['style_tag']}")
            bubble_lines.append(f"核心评价：{praise_info['praise_text']}")
            if praise_info["easter_egg"] is not None:
                bubble_lines.append(f"独特魅力：{praise_info['easter_egg']}")

            # 设定气泡区域的位置和大小（在图像底部预留一块区域）
            margin = 20  # 距离边缘的空隙
            line_height = 24  # 每行文字的高度
            num_lines = len(bubble_lines)

            # 气泡高度：根据行数动态计算，再稍微留一些上下间距
            bubble_height = line_height * num_lines + 2 * margin
            y_bottom = h - margin
            y_top = max(0, y_bottom - bubble_height)
            x_left = margin
            x_right = w - margin

            # 创建一个与图像同尺寸的“遮罩层”，用来绘制半透明矩形
            overlay = vis_image.copy()

            # 画一个填充矩形作为气泡背景（使用淡紫色 / 金黄色等柔和颜色）
            # 这里选用淡紫色调 (BGR: 200, 160, 255)
            cv2.rectangle(
                overlay,
                (x_left, y_top),
                (x_right, y_bottom),
                (200, 160, 255),
                -1,  # -1 表示填充
            )

            # 把 overlay 与原图做加权融合，得到半透明效果
            alpha = 0.6  # 透明度（0~1，越大越不透明）
            vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)

            # 在气泡内部写入文字，使用深色字体，保证可读性
            text_x = x_left + 15
            text_y = y_top + 30  # 第一行文字起始 y

            for line in bubble_lines:
                # 使用中文绘制函数逐行写入文本
                vis_image = cv2_add_chinese_text(
                    vis_image,
                    line,
                    (text_x, text_y),
                    font_size=20,
                    color=(60, 30, 80),  # 深紫色字体（BGR），与背景形成对比
                )
                text_y += line_height

            # ---------- 6. 保存可视化结果到本地文件 ----------
            cv2.imwrite(output_path, vis_image)

        # 返回所有结果（不管是否保存可视化），方便上层做二次处理
        return {
            "three_courts": three_courts,
            "five_eyes": five_eyes,
            "symmetry": symmetry,
            "score": score,
            "style_info": style_info,
            "praise_info": praise_info,
        }


# ===========================
#  脚本直接运行时的测试逻辑
# ===========================
if __name__ == "__main__":
    """
    当我们在命令行中直接运行这个脚本时（例如：python face_logic.py），
    这里的代码会被执行，用于测试和演示效果。
    """

    # 要测试的图片名称，这里约定为“和脚本在同一目录下的 test.jpg”
    test_image = "test.jpg"

    print("开始分析图片：", test_image)

    try:
        # 调用上面编写的主分析函数（默认会生成 visualized_result.jpg）
        result = analyze_face(test_image)

        # 从结果中取出各个部分，方便打印
        three = result["three_courts"]
        five = result["five_eyes"]
        sym = result["symmetry"]
        score = result["score"]

        print("\n===== 三庭（Three Courts）比例 =====")
        print(f"上庭高度: {three['upper']:.2f} 像素, 占总高度比例: {three['upper_ratio']:.3f}")
        print(f"中庭高度: {three['middle']:.2f} 像素, 占总高度比例: {three['middle_ratio']:.3f}")
        print(f"下庭高度: {three['lower']:.2f} 像素, 占总高度比例: {three['lower_ratio']:.3f}")
        print(f"三庭总高度: {three['total']:.2f} 像素")

        print("\n===== 五眼（Five Eyes）比例 =====")
        print(f"左脸到左眼外眼角距离: {five['left_margin']:.2f} 像素, 比例: {five['left_margin_ratio']:.3f}")
        print(f"左眼宽度: {five['left_eye_width']:.2f} 像素, 比例: {five['left_eye_ratio']:.3f}")
        print(f"眼间距: {five['eye_gap']:.2f} 像素, 比例: {five['eye_gap_ratio']:.3f}")
        print(f"右眼宽度: {five['right_eye_width']:.2f} 像素, 比例: {five['right_eye_ratio']:.3f}")
        print(f"右眼外眼角到右脸距离: {five['right_margin']:.2f} 像素, 比例: {five['right_margin_ratio']:.3f}")
        print(f"五眼总宽度: {five['total']:.2f} 像素")

        print("\n===== 左右对称度 =====")
        print(f"估计的脸部中心轴 x 坐标: {sym['center_x']:.2f}")
        print(f"整体对称度评分 (0~1): {sym['symmetry_score']:.3f}")
        print("各部位左右距离对比（数值越接近越对称）:")
        for detail in sym["pair_details"]:
            print(
                f"  {detail['pair']}: "
                f"左距中心={detail['left_distance_to_center']:.2f}, "
                f"右距中心={detail['right_distance_to_center']:.2f}, "
                f"差值={detail['diff']:.2f}"
            )

        print("\n===== 几何颜值评分（仅供娱乐） =====")
        print(f"综合几何分: {score['geometry_score']:.2f} / 100")

        sub = score["sub_scores"]
        print(f"三庭子分 (0~1): {sub['three_courts_score_0_1']:.3f}, 平均偏差: {sub['three_courts_avg_diff']:.3f}")
        print(f"五眼子分 (0~1): {sub['five_eyes_score_0_1']:.3f}, 平均偏差: {sub['five_eyes_avg_diff']:.3f}")
        print(f"对称度子分 (0~1): {sub['symmetry_score_0_1']:.3f}")

        print("\n提示：以上分数仅基于简单几何比例模型，不代表真正的颜值评判，请理性看待。")

    except FileNotFoundError as e:
        # 如果找不到 test.jpg，会给出友好的提示
        print("错误：", str(e))
    except ValueError as e:
        # 如果没有检测到人脸等，也会有提示
        print("错误：", str(e))
    except Exception as e:
        # 捕获其他意外错误，防止程序直接崩溃
        print("发生未知错误：", str(e))
