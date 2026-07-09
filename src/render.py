"""HTML 版型 → 藥袋影像 + 精確 bbox(Playwright / Chromium)。

免手標核心:版型中每個欄位都包 `data-field` 元素、每行文字包 `data-line`,
渲染後直接用 getBoundingClientRect() 取像素座標——標註零人工成本。
device_scale_factor=1,故 CSS 像素 == 輸出影像像素,bbox 不需再換算。
"""
import random
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

_EXTRACT_JS = """
() => {
  const grab = (el) => {
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y),
             w: Math.round(r.width), h: Math.round(r.height) };
  };
  return {
    fields: [...document.querySelectorAll('[data-field]')].map(el => ({
      field: el.dataset.field,
      pii: el.dataset.pii === 'true',
      text: el.innerText.replace(/\\s+/g, ' ').trim(),
      bbox: grab(el),
    })),
    lines: [...document.querySelectorAll('[data-line]')].map(el => ({
      text: el.innerText.replace(/\\s+/g, ' ').trim(),
      bbox: grab(el),
    })),
  };
}
"""


def _clamp(bbox: dict, width: int, height: int) -> dict:
    x = min(max(bbox["x"], 0), width - 1)
    y = min(max(bbox["y"], 0), height - 1)
    w = max(min(bbox["w"], width - x), 1)
    h = max(min(bbox["h"], height - y), 1)
    return {"x": x, "y": y, "w": w, "h": h}


class Renderer:
    """單一 Chromium instance 重複使用,避免每張圖都重開瀏覽器。"""

    def __init__(self, templates_dir: str | Path):
        self._env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html"]))
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._page = self._browser.new_page(device_scale_factor=1)

    def render(self, template_cfg: dict, ctx: dict,
               out_png: Path) -> tuple[tuple[int, int], list[dict], list[dict]]:
        """渲染一張藥袋 → 存 PNG,回傳 (影像尺寸, 欄位 bbox, OCR 行)。"""
        width = template_cfg["width"]
        html = self._env.get_template(template_cfg["file"]).render(**ctx)

        self._page.set_viewport_size(
            {"width": width, "height": template_cfg["min_height"]})
        self._page.set_content(html, wait_until="load")
        height = max(self._page.evaluate(
            "document.documentElement.scrollHeight"), template_cfg["min_height"])
        self._page.set_viewport_size({"width": width, "height": height})

        data = self._page.evaluate(_EXTRACT_JS)
        for item in data["fields"] + data["lines"]:
            item["bbox"] = _clamp(item["bbox"], width, height)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        self._page.screenshot(path=str(out_png))
        return (width, height), data["fields"], data["lines"]

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sample_variation(variation_cfg: dict, rng: random.Random) -> dict:
    """每張藥袋隨機抽版內參數(字體/字級/中英順序/條碼)。"""
    return {
        "font_family": rng.choice(variation_cfg["fonts"]),
        "base_font_px": rng.choice(variation_cfg["base_font_px"]),
        "name_order": rng.choice(variation_cfg["name_order"]),
        "show_barcode": rng.choice(variation_cfg["show_barcode"]),
    }


def drug_display_name(rx: dict, name_order: str) -> str:
    drug = rx["drug"]
    zh, en, st = drug["name_zh"], drug["name_en"], rx["strength"]
    form = drug["form"]
    if name_order == "en_first":
        return f"{en} {st} ({zh}) {form}"
    return f"{zh} {en} {st} {form}"
