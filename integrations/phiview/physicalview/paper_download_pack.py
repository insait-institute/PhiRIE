"""Freeze captured groups into self-contained offline preview and original ZIPs."""

import argparse
import csv
import fcntl
import hashlib
import html
import io
import json
import re
import time
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath


def page(groups, full, stamp):
    counts = Counter(r["dataset"] for r in groups)
    options = "".join(
        f'<option value="{k}">{k} ({v} 组)</option>' for k, v in counts.items()
    )
    keys = sorted({r["feature"] for r in groups})
    feature_options = "".join(f"<option>{k}</option>" for k in keys)
    cards = []
    for r in groups:
        rel = f"{r['dataset']}/{r['scene']}/{r['feature']}"
        detail = ""
        if full:
            links = [
                f'<a target="_blank" href="{rel}/{Path(f).name}">{Path(f).name}</a>'
                for f in r["frames"]
            ]
            links += [
                f'<a target="_blank" href="{rel}/{v}">视频：{v}</a>'
                for v in r["videos"]
            ]
            detail = (
                "<details><summary>查看原图、视频和可编辑面板</summary><nav>"
                + "".join(links)
                + f'<a target="_blank" href="{rel}/panel.svg">可编辑 SVG</a></nav></details>'
            )
        cards.append(
            f'<article data-dataset="{r["dataset"]}" data-feature="{r["feature"]}"><h2>{r["dataset"]} / {r["scene"]}</h2><p>{r["feature"]} · 论文质量：{html.escape(r["review"])}</p><a target="_blank" href="{rel}/review-preview.jpg"><img loading="lazy" src="{rel}/review-preview.jpg" alt="{r["feature"]} 的浏览预览"></a>{detail}</article>'
        )
    kind = "原图完整包" if full else "轻量浏览包（JPEG 预览，不是论文原图）"
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PhiView 离线图库</title>
<style>body{{margin:0;background:#f2f4f7;color:#182230;font:16px system-ui,sans-serif}}header,main{{max-width:1280px;margin:auto;padding:24px}}header{{background:white}}h1{{margin-top:0}}select{{padding:10px;margin:8px 12px 8px 0;max-width:95%}}article{{background:white;border-radius:12px;padding:20px;margin-bottom:24px}}h2{{font-size:18px}}img{{width:100%;height:auto}}p{{line-height:1.6;color:#4b5563}}nav{{display:flex;gap:16px;flex-wrap:wrap;padding:16px 0}}summary{{cursor:pointer;padding:12px 0}}[hidden]{{display:none!important}}</style>
<header><h1>PhiView 离线图库</h1><p>{kind} · 快照：{stamp}<br>共 {len(groups)} 组。截图执行完成不代表论文质量通过；补全残影、生成纹理和机器人可靠操纵仍需改进。</p><p>无需安装或联网。可筛选数据集和功能；点击预览放大。<a href="README.txt">使用说明</a> · <a href="coverage.csv">完整覆盖表</a></p><select id="dataset"><option value="">全部数据集</option>{options}</select><select id="feature"><option value="">全部功能</option>{feature_options}</select><span id="count"></span></header>
<main>{"".join(cards)}</main><script>const ds=document.getElementById('dataset'),ft=document.getElementById('feature');function filter(){{let n=0;document.querySelectorAll('article').forEach(a=>{{a.hidden=!!((ds.value&&a.dataset.dataset!==ds.value)||(ft.value&&a.dataset.feature!==ft.value));if(!a.hidden)n++;}});document.getElementById('count').textContent=n+' 组';}}ds.addEventListener('change',filter);ft.addEventListener('change',filter);filter();</script></html>"""


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(
            v for k, v in attrs if k in ("href", "src") and v and not v.startswith("#")
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    slug = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    with (root / ".paper-pack.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        rows = json.loads((root / "coverage.json").read_text())
        scene_data = {}
        groups = []
        for row in rows:
            if row["status"] != "captured":
                continue
            scene = root / row["dataset"] / row["scene"]
            key = (row["dataset"], row["scene"])
            if key not in scene_data:
                scene_data[key] = (scene / "features.json").read_bytes()
            feature = json.loads(scene_data[key])[row["feature"]]
            assert feature["status"] == "captured" and feature["frames"], row
            folder = scene / row["feature"]
            for rel in feature["frames"]:
                assert (scene / rel).is_file() and (scene / rel).with_suffix(
                    ".json"
                ).is_file(), rel
            assert (folder / "review-preview.jpg").is_file() and (
                folder / "panel.svg"
            ).is_file(), folder
            groups.append(
                dict(
                    row,
                    frames=feature["frames"],
                    videos=sorted(p.name for p in folder.glob("*.mp4")),
                )
            )
        snapshots = {
            "coverage.json": json.dumps(rows, indent=2).encode(),
            "SCENES.md": (root / "SCENES.md").read_bytes()
            if (root / "SCENES.md").exists()
            else b"# Scene inventory\nSee coverage.csv for the captured snapshot.\n",
        }
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=list(rows[0])
            if rows
            else ["dataset", "scene", "feature", "status", "pngs", "review", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)
        snapshots["coverage.csv"] = buf.getvalue().encode()
    counts = Counter(r["dataset"] for r in groups)
    pngs = sum(len(r["frames"]) for r in groups)
    reports = []
    for full in (False, True):
        kind = "full" if full else "preview"
        prefix = f"phiview-{kind}"
        target = a.out / f"phiview-{kind}-{slug}.zip"
        partial = target.with_suffix(".zip.partial")
        manifest = []
        with zipfile.ZipFile(partial, "w", allowZip64=True) as z:

            def add(rel, data):
                assert (
                    not PurePosixPath(rel).is_absolute()
                    and ".." not in PurePosixPath(rel).parts
                ), rel
                compression = (
                    zipfile.ZIP_STORED
                    if Path(rel).suffix.lower() in (".png", ".jpg", ".mp4")
                    else zipfile.ZIP_DEFLATED
                )
                z.writestr(
                    f"{prefix}/{rel}", data, compress_type=compression, compresslevel=3
                )
                manifest.append(
                    {
                        "file": rel,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )

            doc = page(groups, full, stamp).encode()
            add("index.html", doc)
            add("gallery.html", doc)
            readme = f"""PhiView 离线图库\n\n快照：{stamp}\n\n解压整个 ZIP，双击 index.html，用 Chrome、Edge、Firefox 或 Safari 打开。无需 Python、服务器、安装依赖或联网。不要只从 ZIP 内单独打开 HTML。\n\n收录：{dict(counts)}，共 {len(groups)} 组，原始帧总数 {pngs}。\n本包：{"完整 PNG 原图、视频、侧车 JSON、可编辑 SVG 和浏览预览。SVG 通过相对路径引用同目录原图，请保留文件夹结构。" if full else "JPEG 拼图预览。它们不是论文原图；需要无损原图、视频或 SVG 时下载 full 完整包。"}\n\n截图执行完成不等于论文质量通过，本快照中 {sum(r["review"] == "approved" for r in groups)} 组获论文质量验收。补全残影、生成物体纹理和机械臂可靠操纵仍待改进。ScanNet++ 对象建议包含 GT 辅助信息，LIBERO 使用原生 GT 位姿和几何。机器人图不构成可靠抓取或任意命令操纵的验证。导出尺寸不等于输入观测分辨率。\n\n原始 PNG 未后期修图。preview JPG 仅供浏览；完整包 SVG 的布局和文字保留，嵌入 PNG 改为相对文件引用以避免重复存储。\n\ncoverage.csv 记录全部 {len(rows)} 个请求组的快照；本 ZIP 仅收录当时已经 captured 的组。manifest.json 包含包内文件的 SHA-256。后台后续生成的结果不自动更新此 ZIP。\n"""
            add("README.txt", readme.encode())
            for rel, data in snapshots.items():
                add(rel, data)
            for (dataset, scene), data in scene_data.items():
                if full:
                    add(f"{dataset}/{scene}/features.json", data)
            for i, row in enumerate(groups):
                rel = f"{row['dataset']}/{row['scene']}/{row['feature']}"
                folder = root / rel
                if not full:
                    add(
                        rel + "/review-preview.jpg",
                        (folder / "review-preview.jpg").read_bytes(),
                    )
                else:
                    for p in sorted(folder.iterdir()):
                        if not p.is_file() or p.suffix not in (
                            ".png",
                            ".jpg",
                            ".json",
                            ".mp4",
                        ):
                            continue
                        data = p.read_bytes()
                        if p.suffix == ".png":
                            side = json.loads(p.with_suffix(".json").read_text())
                            assert hashlib.sha256(data).hexdigest() == side["sha256"], p
                        add(rel + "/" + p.name, data)
                    layout = json.loads((folder / "panel-layout.json").read_text())
                    names = iter(layout["frames"])
                    svg = (folder / "panel.svg").read_text()
                    found = re.findall(r'href="data:image/png;base64,[^"]+"', svg)
                    assert len(found) == len(layout["frames"]), folder
                    svg = re.sub(
                        r'href="data:image/png;base64,[^"]+"',
                        lambda _: f'href="{html.escape(next(names), quote=True)}"',
                        svg,
                    )
                    add(rel + "/panel.svg", svg.encode())
                if i % 40 == 0:
                    print(f"{kind}: {i + 1}/{len(groups)} groups", flush=True)
            info = {
                "created_utc": stamp,
                "kind": kind,
                "groups": len(groups),
                "datasets": dict(counts),
                "source_pngs": pngs,
                "publication_approved_groups": sum(
                    r["review"] == "approved" for r in groups
                ),
                "files": manifest,
            }
            z.writestr(
                prefix + "/manifest.json",
                json.dumps(info, ensure_ascii=False, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            assert bad is None, bad
            members = set(z.namelist())
            parser = Links()
            parser.feed(z.read(prefix + "/index.html").decode())
            for link in parser.links:
                assert prefix + "/" + link in members, link
            for name in members:
                if name.endswith("/panel.svg"):
                    for link in re.findall(r'href="([^"]+)"', z.read(name).decode()):
                        assert str(PurePosixPath(name).parent / link) in members, (
                            name,
                            link,
                        )
        partial.rename(target)
        h = hashlib.sha256()
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        target.with_suffix(".zip.sha256").write_text(
            h.hexdigest() + "  " + target.name + "\n"
        )
        report = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": h.hexdigest(),
            "groups": len(groups),
            "datasets": dict(counts),
            "source_pngs": pngs,
            "zip_crc_verified": True,
            "offline_links_verified": True,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    (a.out / f"package-report-{slug}.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
