安裝套件:
pip install -r requirements.txt

執行:
python yolo3.py

來源: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

我是拿底下的 yolo3.py 改

修改的部分
* 23行的 ln[i[0] - 1] $\rightarrow$ ln[i - 1]
* 把 82 行註解掉
* 把底下的 createTrackbar 拿掉
* 修改 cv.waitKey(0)，這邊可以看自己方便去調整
