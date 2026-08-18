from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "evaluation" / "hipad_b2d_pnn_control_epoch0150_0725"
OUT = ROOT / "outputs" / "pnn_closed_loop_cases"


CASES = [
    {
        "id": "2091",
        "name": "01_junction_interaction",
        "title": "案例 1｜无信号路口交互：观察横向来车，安全完成左转",
        "glob": "*RouteScenario_2091*",
        "frames": ["0200", "0240", "0280", "0320"],
        "steps": ["接近路口并减速观察", "横向车辆进入冲突区", "保持让行并规划左转", "冲突解除后完成通过"],
        "note": "PNN 的规划轨迹随交通参与者位置动态调整；全程未与横向车辆发生碰撞。",
    },
    {
        "id": "25863",
        "name": "02_pedestrian_avoidance",
        "title": "案例 2｜行人避让：提前制动，待横穿完成后恢复通行",
        "glob": "*RouteScenario_25863*",
        "frames": ["0200", "0240", "0280", "0320"],
        "steps": ["检测路侧行人并开始减速", "多名行人进入车道", "低速等待行人通过", "安全间隙形成后继续行驶"],
        "note": "相机与 BEV 均持续跟踪横穿行人，轨迹由直行推进切换为保守等待，再平滑恢复。",
    },
    {
        "id": "24340",
        "name": "03_out_of_lane_recovery",
        "title": "案例 3｜受扰越界恢复（ControlLoss）：偏移后重新收敛到车道",
        "glob": "*RouteScenario_24340*",
        "frames": ["0000", "0080", "0160", "0240"],
        "steps": ["正常沿车道行驶", "控制扰动引起横向偏移", "规划轨迹主动回正", "重新稳定在目标车道内"],
        "note": "BEV 中可见自车与规划轨迹在扰动后重新对齐道路中心，体现闭环纠偏和恢复能力。",
    },
]


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_case(case):
    route = next(SRC.glob(case["glob"]))
    canvas = Image.new("RGB", (2100, 1500), "#f4f7fb")
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, 2100, 150), fill="#10233f")
    d.text((70, 42), case["title"], font=font(43, True), fill="white")

    for i, (frame, step) in enumerate(zip(case["frames"], case["steps"])):
        x = 55 + (i % 2) * 1020
        y = 185 + (i // 2) * 540
        im = Image.open(route / "images" / f"{frame}.jpg").convert("RGB").resize((990, 270))
        canvas.paste(im, (x, y + 64))
        rounded(d, (x, y, x + 990, y + 55), 12, "#dce8f7")
        d.ellipse((x + 15, y + 9, x + 55, y + 49), fill="#1677ff")
        d.text((x + 28, y + 10), str(i + 1), font=font(24, True), fill="white", anchor="ma")
        d.text((x + 72, y + 10), step, font=font(28, True), fill="#10233f")
        d.text((x + 835, y + 12), f"frame {frame}", font=font(20), fill="#4e6580")
        d.rectangle((x, y + 64, x + 990, y + 334), outline="#8ca3bd", width=2)

    rounded(d, (55, 1295, 2045, 1445), 18, "#e7f5ee", outline="#75b995", width=2)
    d.text((88, 1320), "结论", font=font(30, True), fill="#16734b")
    d.text((185, 1318), case["note"], font=font(28), fill="#173b2c")
    d.text((88, 1385), f"数据源：{route.name}", font=font(19), fill="#526b60")
    canvas.save(OUT / f"{case['name']}.png", quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        make_case(case)


if __name__ == "__main__":
    main()
