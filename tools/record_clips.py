"""Record screen clips of the live status page for the demo video (Playwright, 1920x1080 WebM).

usage: record_clips.py [url]     writes docs/clips/*.webm
"""
import asyncio
import shutil
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "clips"
URL = sys.argv[1] if len(sys.argv) > 1 else "https://d206jwadatehui.cloudfront.net/"
SIZE = {"width": 1920, "height": 1080}


async def clip(browser, name, actions):
    tmp = OUT / f"_{name}"
    # record a 1440x810 viewport at its own size; make_video.py upscales it to 1920x1080, so the page text ends up
    # about 33% larger and readable on video (Playwright pads instead of scaling when the sizes differ)
    view = {"width": 1440, "height": 810}
    ctx = await browser.new_context(viewport=view, record_video_dir=str(tmp), record_video_size=view)
    page = await ctx.new_page()
    await page.goto(URL, wait_until="networkidle")
    await page.wait_for_timeout(1500)
    await actions(page)
    await ctx.close()
    vid = next(tmp.glob("*.webm"))
    shutil.move(str(vid), OUT / f"{name}.webm")
    shutil.rmtree(tmp)
    print("recorded", name)


async def smooth_scroll(page, selector, offset=-70, ms=1800):
    await page.evaluate(
        """([sel, off, ms]) => new Promise(res => {
            const y0 = scrollY, y1 = document.querySelector(sel).getBoundingClientRect().top + scrollY + off;
            const t0 = performance.now();
            const step = (t) => { const k = Math.min(1, (t - t0) / ms); scrollTo(0, y0 + (y1 - y0) * (1 - Math.cos(k * Math.PI)) / 2);
                                  k < 1 ? requestAnimationFrame(step) : res(); };
            requestAnimationFrame(step); })""", [selector, offset, ms])


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()

        async def hero_live(page):
            await page.wait_for_timeout(3000)
            await smooth_scroll(page, "#live")
            await page.wait_for_timeout(5000)
            await page.click("#live details summary")
            await smooth_scroll(page, "#live-trace", -300)
            await page.wait_for_timeout(5000)

        async def replay(page):
            await smooth_scroll(page, "#replay", -64, 1200)
            # cycle 2 is 09:00 CDT, when the agent tightens to 15-min checks; 8 is the blocked request at 10:30
            await page.evaluate("showCycle(2)")
            await page.wait_for_timeout(3000)
            for i in range(3, 16):
                await page.evaluate(f"showCycle({i})")
                await page.wait_for_timeout(3600)
            await page.wait_for_timeout(2000)

        async def evaluation(page):
            await smooth_scroll(page, "#evaluation", -64, 1200)
            await page.wait_for_timeout(5000)
            await smooth_scroll(page, "#evaluation figure", -100, 1500)
            await page.wait_for_timeout(5000)
            await smooth_scroll(page, "#eval-fails", -200, 1500)
            await page.wait_for_timeout(6000)

        async def architecture(page):
            await smooth_scroll(page, "#how", -64, 1200)
            await page.wait_for_timeout(9000)

        # stills of the replay at the cycles the narration talks about (cut on the spoken cue in make_video.py)
        if not sys.argv[2:] or "replay_stills" in sys.argv[2:]:
            ctx = await b.new_context(viewport={"width": 1440, "height": 810})
            page = await ctx.new_page()
            await page.goto(URL, wait_until="networkidle")
            await page.wait_for_timeout(1500)
            await smooth_scroll(page, "#replay", -64, 10)
            for i in (2, 3, 6, 8, 9, 10, 12):
                await page.evaluate(f"showCycle({i})")
                await page.wait_for_timeout(1500)
                await page.screenshot(path=str(OUT / f"replay_{i:02d}.png"))
            await ctx.close()
            print("recorded replay stills")
        only = sys.argv[2:] or ["1_hero_live", "2_replay", "3_evaluation", "4_architecture"]
        for name, fn in [("1_hero_live", hero_live), ("2_replay", replay), ("3_evaluation", evaluation),
                         ("4_architecture", architecture)]:
            if name in only:
                await clip(b, name, fn)
        await b.close()


if __name__ == "__main__":
    asyncio.run(main())
