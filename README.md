# 道路标线识别 / 车道数统计（OpenCV + Python）

输入一张行车视角的照片或一段视频，输出：

* 车道数量（`lane_count`）
* 每条纵向标线的位置、**实线/虚线**、**白/黄**
* **鱼骨线 / 导流线（chevron, hatched area）** 区域，并且不把它算成车道
* 可视化：原图叠加 + 鸟瞰调试图

## 安装

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 使用

```bash
# 单张图片
python -m road_detect.cli image samples/three_lanes.png --out out/vis.png --json out/r.json --bev out/bev.png --lines out/lines.png

# 视频（逐帧检测，取众数作为稳定车道数）
python -m road_detect.cli video drive.mp4 --out out/drive.mp4

# 摄像头实时识别（0 是默认摄像头；也可传手机/IP 摄像头的 rtsp/http 地址）
python -m road_detect.cli camera 0
python -m road_detect.cli camera 0 --window-size 960x540        # 初始窗口大小，之后可鼠标拖拽缩放
python -m road_detect.cli camera 0 --width 1920 --height 1080   # 采集分辨率
python -m road_detect.cli camera 0 --record out/cam.mp4         # 同时录像
python -m road_detect.cli camera rtsp://192.168.1.20:554/live   # 网络摄像头
# 默认会额外弹出 "lane lines (extracted)" 窗口，显示提取出的分割线（白/黄、实/虚编号）
# 和紫色的鱼骨/导流区；不需要可加 --no-lines 或按 l 关闭
# 窗口快捷键：q/ESC 退出，s 截图（同时存一张分割线图），l 分割线窗口，b 鸟瞰调试图，r 显示 ROI 梯形
# 所有窗口都是可自由缩放的（WINDOW_NORMAL）
# 无图形界面（服务器 / WSL 无 GUI）时加 --headless，只打印 JSON

# 调 ROI：把梯形画到图上，改 config 直到梯形正好套住路面
python -m road_detect.cli roi your_road.jpg --out out/roi.png --config my.json
```

输出示例：

```json
{
  "lane_count": 2,
  "lanes": [{"index": 1, "left_x_m": 0.76, "right_x_m": 4.25, "width_m": 3.49, "inferred": false}],
  "lines": [{"x_m": 0.76, "style": "solid", "color": "yellow", "fill_ratio": 1.0}],
  "hatch_zones": [{"stripes": 4, "angle_deg": 68.3, "x_range_m": [7.9, 11.1], "y_range_m": [17.4, 29.1]}]
}
```

代码里调用：

```python
import cv2
from road_detect import Config, RoadMarkingDetector

result = RoadMarkingDetector(Config()).detect(cv2.imread("road.jpg"))
print(result.lane_count, [l.style for l in result.lines])
```

## 算法流程

1. **逆透视变换（鸟瞰图）** `perspective.py`
   ROI 梯形 → 矩形。鸟瞰图里车道线接近竖直，这是后面所有判断的基础。
2. **标线提取** `markings.paint_mask`
   HLS 颜色阈值（白：高亮度低饱和；黄：色相 15–38）+ 水平方向 **top-hat**。
   top-hat 只保留“比左右邻域亮的细结构”，因此对阴影、路面偏色、旧漆比纯阈值稳。
3. **鱼骨线/导流线识别** `markings.find_hatch_zones`
   先用竖直核形态学开运算去掉车道线（否则斜条会和边线连成一块），
   剩下的连通域按：长宽比、与行车方向的夹角（默认 >22°）、最小长度、
   比路面亮多少、条纹长度是否相近、间距是否规则、整体宽度是否合理 逐项筛选，
   ≥3 条同向且相邻的斜条 → 判为导流区，并从掩膜中删除，**不参与车道计数**。
4. **车道线跟踪** `lanes.find_lane_lines`
   底半幅列直方图找峰 → 滑窗上溯 → 二次多项式拟合；
   按“有标线的行占比”判实线/虚线，按 HLS 采样判白/黄；过宽的结构（护栏、路缘、反光）剔除。
5. **车道计数** `lanes.count_lanes`
   相邻两条线的间距：
   * < 2.2 m：双黄线/双白线，不算车道；
   * 2.2–4.6 m：一条车道；
   * 更宽：按 3.5 m 一条切分（标线磨损或被车挡住时仍能补回，结果标 `inferred: true`）；
   * 落在导流区里的间隔：不算车道。

## 标定（最重要的一步）

像素与米的换算来自 ROI 梯形。默认开启 `auto_scale`：
把**相邻标线间距的中位数**当作一条标准车道（3.5 m）来反推比例尺，
所以 ROI 画得不太准也能得到正确车道数。

如果你的相机已经标定好，把 `auto_scale` 设为 `false`，
并按实际情况填 `bev_width_m` / `bev_length_m`，此时输出的米制数值才是真实距离。

自定义参数：把 `Config` 的字段写成 JSON 传给 `--config`，例如

```json
{"roi_top_y": 0.60, "roi_bottom_left_x": 0.08, "roi_bottom_right_x": 0.92, "nominal_lane_width_m": 3.75}
```

## 测试

```bash
python -m pytest -q          # 9 个用例：合成场景 1/2/3 车道、导流区、实虚线、计数规则
python tools/make_synthetic.py --outdir samples   # 重新生成合成样图
```

`tests/` 用 `tools/make_synthetic.py` 生成带真值的合成路面（含导流区），
车道数与真值逐一比对。真实照片（`samples/real/`，Udacity 公开测试图）
在默认参数下也能正确给出 ROI 内的车道数。

## 已知限制

* 只统计 **ROI 梯形覆盖范围内** 的车道；弯道半径小或相邻车道在画面外时会少数。
* 大面积阴影、积水反光、雪天会明显掉点；夜间需要重新调 `white_l_min` / `tophat_thresh`。
* 纯几何方法，没有语义分割；被车辆压住的标线靠“宽间隔切分”补，不保证正确。
  若要工业级鲁棒性，建议在这套输出上再叠一个分割模型（如 LaneNet / UFLD），
  本仓库的后处理（导流区剔除、间距成车道）可以直接复用。
