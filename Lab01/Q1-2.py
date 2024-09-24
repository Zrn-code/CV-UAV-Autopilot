import cv2
import numpy as np

# 调整对比度和亮度的函数
def adjust_contrast_brightness(image, contrast, brightness):
    # 将图像转换为 int32 以防止溢出
    new_image = (image.astype(np.int32) - 127) * (contrast / 127 + 1) + 127 + brightness
    # 使用 np.clip 防止溢出，并转换回 uint8
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image

# 读取图片
image = cv2.imread('test.jpg')



# 定义对比度和亮度调整参数
contrast = 50    # 对比度调整量，可根据需要调整
brightness = 30  # 亮度调整量，可根据需要调整

# 识别蓝点的掩码
# 条件: B > 100 且 B * 0.6 > G 且 B * 0.6 > R
blue_mask = (image[:, :, 0] > 100) & (image[:, :, 0] * 0.6 > image[:, :, 1]) & (image[:, :, 0] * 0.6 > image[:, :, 2])

# 识别黄点的掩码
# 黄点通常由高 R 和 G 值组成，且相对较低的 B 值
# 我们可以定义黄点条件为: R > 100 且 G > 100 且 (R + G) / 2 > B
yellow_mask = (image[:, :, 2] > 100) & (image[:, :, 1] > 100) & ((image[:, :, 2] + image[:, :, 1]) / 2 > image[:, :, 0])

# 创建结果图像的副本
result = image.copy()

# 对蓝点进行对比度与亮度调整
result[blue_mask] = adjust_contrast_brightness(result[blue_mask], contrast, brightness)

# 对黄点进行对比度与亮度调整
result[yellow_mask] = adjust_contrast_brightness(result[yellow_mask], contrast, brightness)

# 显示处理后的图像
cv2.imshow('Adjusted Blue and Yellow Points', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存处理后的图像
# cv2.imwrite('adjusted_test.jpg', result)
