import vtracer
import os
from tqdm import tqdm
from multiprocessing import Pool
import cairosvg
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import optuna
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='image/image', help='Directory containing input images')
parser.add_argument('--save_dir', type=str, default='submit', help='Directory to save SVG files')
parser.add_argument('--temp_dir', type=str, default='render', help='Temporary directory for rendered images')
parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for each image')
parser.add_argument('--processes', type=int, default=16, help='Number of processes to use')
args = parser.parse_args()


optuna.logging.set_verbosity(optuna.logging.WARNING)  # 只保留警告及以上


image_dir = args.image_dir
save_dir =  args.save_dir
temp_dir =  args.temp_dir
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)


def check_svg_validity(svg_path):
    """判断SVG是否有效且字符数限制"""
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if len(content) > 50000:
        return False
    if '<svg' not in content:
        return False
    return True


def render_svg_to_png(svg_path, png_path):
    """用cairosvg将svg渲染为png"""
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return True


def calculate_ssim_between_images(img_path1, img_path2):
    """计算两张图片的SSIM"""
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    # # 调整大小一致
    # if arr1.shape != arr2.shape:
    #     from skimage.transform import resize
    #     arr2 = resize(arr2, arr1.shape, anti_aliasing=True)
    score = ssim(arr1, arr2, channel_axis=-1)
    return score
    

# def convert_image_to_svg(file):
#     input_path = os.path.join(image_dir, file)
#     output_path = os.path.join(save_dir, file.replace('.jpg', '.svg'))

#     # 转换为 SVG
#     vtracer.convert_image_to_svg_py(
#         input_path,
#         output_path,
#         colormode='color',        # 多彩模式，可改成 'binary'
#         hierarchical='stacked',   # 分层方式，可改 'cutout'
#         mode='spline',            # 曲线模式，可改 'polygon'
#         filter_speckle=4,
#         color_precision=6,
#         layer_difference=16,
#         corner_threshold=60,
#         length_threshold=4.0,
#         max_iterations=10,
#         splice_threshold=45,
#         path_precision=3
#     )


def convert_and_score(params, input_path, file):
    """用给定参数转换单张图片，返回评分"""
    svg_path = os.path.join(save_dir, file.replace('.jpg', '.svg'))
    png_path = os.path.join(temp_dir, file.replace('.jpg', '_render.png'))

    vtracer.convert_image_to_svg_py(
        input_path,
        svg_path,
        colormode='color',
        hierarchical=params['hierarchical'],
        mode=params['mode'],
        filter_speckle=params['filter_speckle'],
        color_precision=params['color_precision'],
        layer_difference=params['layer_difference'],
        corner_threshold=params['corner_threshold'],
        length_threshold=params['length_threshold'],
        max_iterations=params['max_iterations'],
        splice_threshold=params['splice_threshold'],
        path_precision=params['path_precision']
    )

    if not check_svg_validity(svg_path):
        return 0

    if not render_svg_to_png(svg_path, png_path):
        return 0

    return calculate_ssim_between_images(input_path, png_path)


# with Pool(processes=None) as p:  # 进程池，并行数量缺省为CPU核心数
#     files = [f for f in os.listdir(image_dir)]
#     for _ in tqdm(p.imap_unordered(convert_image_to_svg, files), total=len(files)):
#         pass  # tqdm 进度条更新


def search_best_params_for_image(file):
    input_path = os.path.join(image_dir, file)

    def objective(trial):
        params = {
            'hierarchical': trial.suggest_categorical('hierarchical', ['stacked', 'cutout']),
            'mode': trial.suggest_categorical('mode', ['spline', 'polygon', 'pixel']),
            'filter_speckle': trial.suggest_int('filter_speckle', 0, 10),
            'color_precision': trial.suggest_int('color_precision', 1, 8),
            'layer_difference': trial.suggest_int('layer_difference', 1, 50),
            'corner_threshold': trial.suggest_int('corner_threshold', 0, 100),
            'length_threshold': trial.suggest_float('length_threshold', 3.5, 10.0),
            'max_iterations': trial.suggest_int('max_iterations', 1, 20),
            'splice_threshold': trial.suggest_int('splice_threshold', 0, 100),
            'path_precision': trial.suggest_int('path_precision', 1, 10),
        }
        return convert_and_score(params, input_path, file)

    study = optuna.create_study(direction='maximize')  # 优化SSIM和SVG文件大小
    study.optimize(objective, n_trials=args.n_trials)  # 每张图片20次搜索

    svg_path = os.path.join(save_dir, file.replace('.jpg', '.svg'))
    png_path = os.path.join(temp_dir, file.replace('.jpg', '_render.png'))
    vtracer.convert_image_to_svg_py(
        input_path,
        svg_path,
        colormode='color',
        **study.best_params
    )

    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    render_svg_to_png(svg_path, png_path)

    assert study.best_value == calculate_ssim_between_images(input_path, png_path)
    print(f"Processed {file}: Best SSIM = {study.best_value}, SVG Size = {len(content)} bytes")


# for f in os.listdir(image_dir):
#     search_best_params_for_image(f)
    
with Pool(processes=args.processes) as p:  # 进程池，并行数量缺省为CPU核心数
    files = [f for f in os.listdir(image_dir)]
    for _ in tqdm(p.imap_unordered(search_best_params_for_image, files), total=len(files)):
        pass  # tqdm 进度条更新