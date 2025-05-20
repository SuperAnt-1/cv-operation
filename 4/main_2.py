import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# ---------- 配置 ----------
# 原始图像目录（320×240 灰度图）
DATA_DIR    = '/home/yanai-lab/ma-y/work/assignment/assignment_4/input/2'
# 重构结果保存根目录
BASE_OUT    = '/home/yanai-lab/ma-y/work/assignment/assignment_4/output/2'
# 确保输出目录存在
os.makedirs(BASE_OUT, exist_ok=True)

# ---------- 读取并展平图像 ----------
imgs = []
fnames = []
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.lower().endswith('.jpg'):
        path = os.path.join(DATA_DIR, fname)
        # 打开并转换为 320×240 灰度图
        im = Image.open(path).convert('L').resize((320,240))
        arr = np.asarray(im, dtype=np.float64).flatten()  # 展平到 1×76800
        imgs.append(arr)
        fnames.append(os.path.splitext(fname)[0])
if len(imgs) < 1:
    raise RuntimeError(f"未找到任何 .jpg 文件，请检查 {DATA_DIR}")
X = np.stack(imgs, axis=0)  # 形状 (n_samples, 76800)

# ---------- PCA 压缩与重构并保存（拼接原图和重构图） ----------
for k in [3, 5, 10, 20, 30]:
    # 创建输出子目录 recon_{k}dim
    out_dir = os.path.join(BASE_OUT, f'recon_{k}dim')
    os.makedirs(out_dir, exist_ok=True)

    # PCA 压缩与重构
    pca = PCA(n_components=k, svd_solver='full')
    X_reduced = pca.fit_transform(X)               # 压缩到 k 维
    X_recon   = pca.inverse_transform(X_reduced)   # 重构回 76800 维

    for i, vec in enumerate(X_recon):
        # --- 原图 ---
        orig_im = Image.fromarray(
            np.asarray(
                Image.open(os.path.join(DATA_DIR, fnames[i] + '.jpg'))
                .convert('L').resize((320,240))
            ),
            mode='L'
        )

        # --- 重构图 ---
        img_arr = vec.reshape((240,320))
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        recon_im = Image.fromarray(img_arr, mode='L')

        # --- 拼接左右两幅图 ---
        combined = Image.new('L', (320*2, 240))
        combined.paste(orig_im,   (0,   0))
        combined.paste(recon_im,  (320, 0))

        # --- 保存拼接图 ---
        out_path = os.path.join(out_dir, f'{fnames[i]}_compare_{k}dim.png')
        combined.save(out_path)

    print(f"[{k} 维重构完成] 拼接图已保存到：{out_dir}")
