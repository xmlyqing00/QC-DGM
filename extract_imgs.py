import os
import cv2

in_dir = 'affine_1.0'
opt_num = 3
models = ['QCDGM', 'quad_sinkhorn']
step = 10
offset = 1
total_num = 40

out_dir = os.path.join(in_dir, in_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for model_name in models:
    for opt in range(opt_num):
        subset = os.path.join(in_dir, f'{model_name}_{opt}')
        for i in range(total_num):
            idx = i * step + offset
            in_img_path = os.path.join(subset, f'{idx:04}.png')
            out_img_path = os.path.join(out_dir, f'{idx:04}_{model_name}_{opt}.jpg')
            img = cv2.imread(in_img_path)
            cv2.imwrite(out_img_path, img)
            print(in_img_path, out_img_path)

models = ['vgg16_cie', 'vgg16_ipca', 'vgg16_pca']
think_dir = os.path.join('/Ship03/Sources/FeatureMatching/ThinkMatch', in_dir)
for model_name in models:
    subset = os.path.join(think_dir, f'{model_name}')
    for i in range(total_num):
        idx = i * step + offset
        in_img_path = os.path.join(subset, f'{idx:04}.png')
        out_img_path = os.path.join(out_dir, f'{idx:04}_{model_name}.jpg')
        img = cv2.imread(in_img_path)
        cv2.imwrite(out_img_path, img)
        print(in_img_path, out_img_path)
