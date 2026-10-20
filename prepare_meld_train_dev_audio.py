from pathlib import Path, PurePosixPath
import csv
import re
import tarfile
import tempfile
import subprocess
import shutil
import os
import wave

base = Path("/code/EMART/dataset/MELD.Raw")
pattern = re.compile(r"^dia\d+_utt\d+\.mp4$")

if not shutil.which("ffmpeg"):
    raise SystemExit("错误：找不到 ffmpeg")

def valid_wav(path):
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getframerate() == 16000
                and w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getnframes() > 0
            )
    except Exception:
        return False

def prepare(split):
    print(f"\n========== {split.upper()} ==========", flush=True)

    csv_path = base / f"{split}_sent_emo.csv"
    archive_path = base / f"{split}.tar.gz"
    output_dir = base / f"{split}_wav"

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    expected = {
        f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
        for r in rows
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = []
    seen = set()

    # 只保存当前正在处理的一个视频
    with tempfile.TemporaryDirectory(
        prefix=f"meld_{split}_",
        dir=output_dir
    ) as temp_dir:

        video_path = Path(temp_dir) / "input.mp4"
        wav_temp = Path(temp_dir) / "output.wav"

        # 流式读取压缩包，避免整体解压
        with tarfile.open(archive_path, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                filename = PurePosixPath(member.name).name

                if not pattern.fullmatch(filename):
                    continue

                uid = filename[:-4]

                if uid not in expected or uid in seen:
                    continue

                seen.add(uid)
                output_path = output_dir / f"{uid}.wav"

                if valid_wav(output_path):
                    skipped += 1
                    continue

                try:
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError("无法读取视频")

                    with source, video_path.open("wb") as f:
                        shutil.copyfileobj(source, f, 1024 * 1024)

                    cmd = [
                        "ffmpeg", "-y", "-nostdin",
                        "-hide_banner", "-loglevel", "error",
                        "-i", str(video_path),
                        "-vn",
                        "-ar", "16000",
                        "-ac", "1",
                        "-c:a", "pcm_s16le",
                        "-f", "wav",
                        str(wav_temp),
                    ]

                    result = subprocess.run(
                        cmd, capture_output=True, text=True
                    )

                    if result.returncode != 0:
                        raise RuntimeError(result.stderr[-500:])

                    if not valid_wav(wav_temp):
                        raise RuntimeError("输出 WAV 验证失败")

                    os.replace(wav_temp, output_path)
                    converted += 1

                except Exception as e:
                    failed.append((uid, str(e)))
                    print(f"失败: {uid} - {e}", flush=True)

                finally:
                    video_path.unlink(missing_ok=True)
                    wav_temp.unlink(missing_ok=True)

                if (converted + skipped) % 500 == 0:
                    print(
                        f"进度: 已转换 {converted}, "
                        f"跳过 {skipped}, 失败 {len(failed)}",
                        flush=True
                    )

    missing_video = expected - seen
    invalid = [
        uid for uid in expected
        if not valid_wav(output_dir / f"{uid}.wav")
    ]

    print(f"\n{split.upper()} 最终统计")
    print("CSV 样本数:", len(expected))
    print("压缩包找到:", len(seen))
    print("本次转换成功:", converted)
    print("跳过有效 WAV:", skipped)
    print("转换失败:", len(failed))
    print("压缩包缺失:", len(missing_video))
    print("最终无效/缺失 WAV:", len(invalid))
    print("缺失视频示例:", sorted(missing_video)[:10])
    print("无效 WAV 示例:", sorted(invalid)[:10])
    print("失败示例:", failed[:10])

    return len(invalid) == 0

results = {}
for split in ["train", "dev"]:
    results[split] = prepare(split)

print("\n========== MELD 音频准备结果 ==========")
for split, success in results.items():
    print(f"{split}: {'完成' if success else '仍有问题'}")

if not all(results.values()):
    raise SystemExit(1)
