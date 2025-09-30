import cv2
import imageio

# 读取GIF
gif = imageio.mimread('../training_phase_even.gif')

# 获取GIF信息
height, width, _ = gif[0].shape
fps = 10  # 可以调整帧率

# 创建视频写入器
video = cv2.VideoWriter('training_phase_even.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 逐帧写入
for frame in gif:
    # 将RGB转换为BGR（OpenCV使用BGR格式）
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame_bgr)

video.release()