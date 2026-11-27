"""Record the Gradio app (must be running on 127.0.0.1:7860) on the three examples with Playwright recordVideo -> demo.webm."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent; URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport=dict(width=1280, height=720), record_video_dir=str(OUT / "_video"),
                        record_video_size=dict(width=1280, height=720))
    pg = ctx.new_page(); pg.goto(URL); pg.wait_for_selector("#run"); time.sleep(2)
    for k in range(3):
        pg.locator("tr.tr-body").nth(k).click()
        time.sleep(1.2); pg.click("#run")
        pg.wait_for_function("document.body.innerText.includes('watermarked area')", timeout=30000); time.sleep(3.5)
    vid = pg.video.path(); ctx.close(); b.close()
Path(vid).replace(OUT / "_demo.webm"); print("video", OUT / "_demo.webm")
