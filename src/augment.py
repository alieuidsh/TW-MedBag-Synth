"""資料增強 pipeline + bbox 幾何同步變換。

以 OpenCV/Pillow 實作,效果以名稱 registry 組織,方便自行替換或擴充。

規則:
- 光度類(亮度/對比/熱感紙褪色/反光/雜訊/JPEG/手震模糊):bbox 不變。
- 幾何類(透視/小角度旋轉):回傳 3x3 homography,
  呼叫端對 bbox 四角套同一矩陣後重算外接框——否則標註會對不上。

注意:Windows 中文路徑下 cv2.imread/imwrite 會靜默失敗,
一律使用 imread_u / imwrite_u(np.fromfile + imdecode)。
"""
import random
from pathlib import Path

import cv2
import numpy as np


# ── unicode-safe I/O ─────────────────────────────────────
def imread_u(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"無法讀取影像:{path}")
    return img


def imwrite_u(path: str | Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise IOError(f"無法編碼影像:{path}")
    buf.tofile(str(path))


# ── 光度類(bbox 不變)────────────────────────────────────
def brightness(img: np.ndarray, rng: random.Random) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=rng.uniform(-40, 30))


def contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    a = rng.uniform(0.72, 1.25)
    return cv2.convertScaleAbs(img, alpha=a, beta=127 * (1 - a))


def thermal_fade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """熱感紙褪色:墨色整體向白衰減 + 低頻斑駁不均。"""
    h, w = img.shape[:2]
    fade = rng.uniform(0.25, 0.55)
    blotch = cv2.resize(
        np.random.default_rng(rng.getrandbits(32)).random((6, 4)).astype(np.float32),
        (w, h), interpolation=cv2.INTER_CUBIC)
    keep = 1.0 - fade * (0.6 + 0.4 * blotch)          # 每像素保留的墨量
    ink = 255.0 - img.astype(np.float32)              # 墨 = 距白的深度
    return (255.0 - ink * keep[..., None]).clip(0, 255).astype(np.uint8)


def glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """局部反光:橢圓形白色漸層(模擬護貝/塑膠袋面反光)。"""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    rx, ry = rng.uniform(0.18, 0.4) * w, rng.uniform(0.12, 0.3) * h
    y, x = np.ogrid[:h, :w]
    d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    weight = np.clip(1.0 - d, 0, 1) ** 2 * rng.uniform(0.45, 0.8)
    out = img.astype(np.float32)
    return (out + (255.0 - out) * weight[..., None]).clip(0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(4, 12)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, sigma, img.shape)
    return (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def jpeg_artifact(img: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(25, 55)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def motion_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """手震模糊:隨機方向的線狀 kernel(手持翻拍最常見)。"""
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel = cv2.warpAffine(
        kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                        rng.uniform(0, 180), 1.0), (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


# ── 幾何類(回傳 homography,bbox 需同步變換)─────────────
def perspective(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    jx, jy = 0.025 * w, 0.025 * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, jx), rng.uniform(0, jy)],
                      [w - rng.uniform(0, jx), rng.uniform(0, jy)],
                      [w - rng.uniform(0, jx), h - rng.uniform(0, jy)],
                      [rng.uniform(0, jx), h - rng.uniform(0, jy)]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), borderValue=(255, 255, 255))
    return out, H


def rotate(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3.5, 3.5), 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    return out, np.vstack([M, [0, 0, 1]]).astype(np.float64)


PHOTOMETRIC = {
    "brightness": brightness, "contrast": contrast, "thermal_fade": thermal_fade,
    "glare": glare, "gaussian_noise": gaussian_noise,
    "jpeg_artifact": jpeg_artifact, "motion_blur": motion_blur,
}
GEOMETRIC = {"perspective": perspective, "rotate": rotate}


def transform_bbox(bbox: dict, H: np.ndarray, width: int, height: int) -> dict:
    """bbox 四角套 homography → 取外接框 → 裁到影像內。"""
    pts = np.float32([[bbox["x"], bbox["y"]],
                      [bbox["x"] + bbox["w"], bbox["y"]],
                      [bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]],
                      [bbox["x"], bbox["y"] + bbox["h"]]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    x0 = int(min(max(x0, 0), width - 1))
    y0 = int(min(max(y0, 0), height - 1))
    return {"x": x0, "y": y0,
            "w": max(int(min(x1, width) - x0), 1),
            "h": max(int(min(y1, height) - y0), 1)}


def make_variant(img: np.ndarray,
                 rng: random.Random) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """產生一個增強變體:0~1 個幾何 + 1~3 個光度,回傳 (影像, homography, 名稱)。"""
    names: list[str] = []
    H = None
    out = img
    if rng.random() < 0.5:
        name = rng.choice(list(GEOMETRIC))
        out, H = GEOMETRIC[name](out, rng)
        names.append(name)
    for name in rng.sample(list(PHOTOMETRIC), rng.randint(1, 3)):
        out = PHOTOMETRIC[name](out, rng)
        names.append(name)
    return out, H, names
