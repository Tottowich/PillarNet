import cv2, glob
from pathlib import Path

sequence_path = '/home/sgs/Pictures/second/'
out_name = "/home/sgs/second2.mp4"

images_list = glob.glob(sequence_path + "*.png")
# images_list.sort()

image_ori = cv2.imread(images_list[0], cv2.IMREAD_COLOR)
prefix = Path(images_list[0]).parent

video_size = (image_ori.shape[1]//3, image_ori.shape[0]//3)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(out_name,  fourcc, 5, video_size, True)

name = 'Ground Truth'
font = cv2.FONT_HERSHEY_SIMPLEX


for idx in range(240, 320):
    img = prefix / (str(idx) + '.png')
    img = str(img)
    frame = cv2.imread(img, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, video_size)

    # cv2.imshow("image", frame)
    # cv2.waitKey(0)
    # cv2.putText(frame, name, (0, 10), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)

    video.write(frame)
video.release()
