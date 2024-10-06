import cv2

# 创建一个空列表来存储四个点的坐标
points = []

# 鼠标回调函数，负责记录用户点击的坐标
def get_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
        points.append((x, y))  # 保存点击的坐标
        print(f"Point {len(points)}: ({x}, {y})")
        # 在图像上标记出用户点击的点
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", param)

# 读取目标图片
dst_img = cv2.imread('screen.jpg')

# 设置窗口的大小
window_width = 800  # 窗口宽度
window_height = 600  # 窗口高度

# 创建一个窗口，并设置为可调整大小
cv2.namedWindow("Select Points", cv2.WINDOW_NORMAL)

# 设置窗口大小（锁定）
cv2.resizeWindow("Select Points", window_width, window_height)

# 显示图片
cv2.imshow("Select Points", dst_img)

# 设置鼠标回调函数
cv2.setMouseCallback("Select Points", get_points, dst_img)

# 等待用户选取四个点
print("请点击图片上的四个点，按任意键结束。")
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()

# 输出四个点的坐标
if len(points) == 4:
    print("四个点的坐标为：", points)
else:
    print(f"你只选中了 {len(points)} 个点，请确保选择四个点。")
