# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# import hdf5storage
#
# def visualize_wavelet(image, wavelet='haar', level=1):
#     # 小波分解
#     import pywt
#     import matplotlib.pyplot as plt
#     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
#     arr, slices = pywt.coeffs_to_array(coeffs)
#
#     # 可视化
#     plt.figure(figsize=(8, 8))
#     plt.imshow(arr, cmap='gray')
#     plt.title(f'Wavelet Decomposition ({wavelet})')
#     plt.axis('off')
#     plt.show()
#
#     return arr, coeffs
#
# mat = hdf5storage.loadmat("/data/liumengzu/data/Dataset/ARAD_testing/test_mosaic/ARAD_1K_0901.raw.mat")
# dummy_image = mat['mosaic']
# # norm_factor = mat['norm_factor']
# # 示例：随机图像
# # dummy_image = np.random.rand(128, 128)"/data/liumengzu/data/Dataset/ARAD_testing/test_mosaic/ARAD_1K_0901.raw.mat"
# wavelet_result, coeffs = visualize_wavelet(dummy_image)

import numpy as np
import pywt
import matplotlib.pyplot as plt
import hdf5storage


# def visualize_wavelet_high_freq_only_reconstruction(image, wavelet='haar', level=1):
#     # 小波分解
#     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
#
#     # 构造一个全零的 LL 分量（同 shape）
#     LL = np.zeros_like(coeffs[0])
#
#     # 保留高频分量
#     high_freqs = coeffs[1:]
#
#     # 构建新的系数：LL 为 0，其余保持不变
#     coeffs_high_only = [LL] + high_freqs
#
#     # 执行 IWT 重建图像
#     reconstructed = pywt.waverec2(coeffs_high_only, wavelet=wavelet)
#
#     # 可视化原图和仅用高频分量恢复的图像
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(reconstructed, cmap='gray')
#     plt.title('Reconstructed from HL, LH, HH only')
#     plt.axis('off')
#     plt.show()
#
#     return reconstructed
#
# def visualize_wavelet_LL_only_reconstruction(image, wavelet='haar', level=1):
#     # 小波分解
#     coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
#
#     # 保留 LL，其他置为 0
#     LL, high_freqs = coeffs[0], coeffs[1:]
#     zero_high_freqs = []
#     for detail in high_freqs:
#         # detail 是 (LH, HL, HH)
#         zero_detail = tuple(np.zeros_like(d) for d in detail)
#         zero_high_freqs.append(zero_detail)
#
#     # 构建新的系数，只保留 LL 分量
#     coeffs_zeroed = [LL] + zero_high_freqs
#
#     # 执行 IWT 重建图像
#     reconstructed = pywt.waverec2(coeffs_zeroed, wavelet=wavelet)
#
#     # 可视化原图和只用LL恢复的图像
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(reconstructed, cmap='gray')
#     plt.title('Reconstructed from LL only')
#     plt.axis('off')
#     plt.show()
#
#     return reconstructed
#
# # 载入图像
# mat = hdf5storage.loadmat("/data/liumengzu/data/Dataset/ARAD_testing/test_mosaic/ARAD_1K_0901.raw.mat")
# dummy_image = mat['mosaic']
#
# # 可视化 LL-only 重建图像
# # reconstructed = visualize_wavelet_LL_only_reconstruction(dummy_image)
# reconstructed = visualize_wavelet_high_freq_only_reconstruction(dummy_image)

def reconstruct_ll_and_high_only(image, wavelet='haar', level=1):
    # 小波分解
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # 拆分低频和高频
    LL = coeffs[0]
    high_freqs = coeffs[1:]

    # 构建 LL-only 系数（高频为0）
    zero_high_freqs = [(np.zeros_like(h), np.zeros_like(v), np.zeros_like(d)) for (h, v, d) in high_freqs]
    coeffs_ll_only = [LL] + zero_high_freqs
    image_ll_only = pywt.waverec2(coeffs_ll_only, wavelet=wavelet)

    # 构建 High-only 系数（LL为0）
    zero_LL = np.zeros_like(LL)
    coeffs_high_only = [zero_LL] + high_freqs
    image_high_only = pywt.waverec2(coeffs_high_only, wavelet=wavelet)

    # 相加结果
    image_sum = image_ll_only + image_high_only

    return image_ll_only, image_high_only, image_sum

def reconstruct_ll_and_ll(image, wavelet='haar', level=1):
    # 小波分解
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # 拆分低频和高频
    LL = coeffs[0]
    high_freqs = coeffs[1:]

    # 构建 LL-only 系数（高频为0）
    zero_high_freqs = [(np.zeros_like(h), np.zeros_like(v), np.zeros_like(d)) for (h, v, d) in high_freqs]
    coeffs_ll_only = [LL] + zero_high_freqs
    image_ll_only = pywt.waverec2(coeffs_ll_only, wavelet=wavelet)

    image_ll = pywt.waverec2([LL] + [(LL, LL, LL)], wavelet=wavelet)

    # 构建 High-only 系数（LL为0）
    zero_LL = np.zeros_like(LL)
    coeffs_high_only = [zero_LL] + high_freqs
    image_high_only = pywt.waverec2(coeffs_high_only, wavelet=wavelet)

    # 相加结果
    image_sum = image_ll_only + image_high_only

    return image_ll_only, image_ll, image_high_only, image_sum

# 加载图像
mat = hdf5storage.loadmat("/data/liumengzu/data/Dataset/ARAD_testing/test_mosaic/ARAD_1K_0901.raw.mat")
dummy_image = mat['mosaic']

# 进行重构
ll_img, lll_img, high_img, sum_img = reconstruct_ll_and_ll(dummy_image)

plt.figure(figsize=(25, 5))
plt.subplot(1, 5, 1)
plt.imshow(dummy_image, cmap='gray')
plt.title('ori')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(lll_img, cmap='gray')
plt.title('LL Only')
plt.axis('off')

# 可视化
# plt.figure(figsize=(20, 5))
plt.subplot(1, 5, 3)
plt.imshow(ll_img, cmap='gray')
plt.title('LL Only')
plt.axis('off')

plt.subplot(1, 5 , 4)
plt.imshow(high_img, cmap='gray')
plt.title('HL+LH+HH Only')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(sum_img, cmap='gray')
plt.title('LL + High (Reconstructed)')
plt.axis('off')
plt.tight_layout()
plt.show()
