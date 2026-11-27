"""Record the local Gradio app at 1280 by 720."""

import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUT = Path(__file__).resolve().parent

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    size = {"width": 1280, "height": 720}
    context = browser.new_context(viewport=size, record_video_dir=OUT / "_video", record_video_size=size)
    page = context.new_page()
    page.goto("http://127.0.0.1:7860")
    page.locator("#demo-examples img").first.click()
    expect(page.locator("#demo-input img")).to_have_attribute("src", re.compile(r"edited\.png"))
    page.locator("#run").click()
    expect(page.locator("#info")).to_contain_text("predicted watermarked cells", timeout=15000)
    if screenshot := os.getenv("SHOWCASE_SCREENSHOT"):
        page.screenshot(path=screenshot)
    page.wait_for_timeout(3000)
    video = page.video
    context.close()
    if video is None:
        raise RuntimeError("No recording produced.")
    Path(video.path()).replace(OUT / "_demo.webm")
    browser.close()
