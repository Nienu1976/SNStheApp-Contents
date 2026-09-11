"""
文脈内アクセント検証ツール (プロトタイプ)

使い方:
    python3 check_context_accent.py <CSVファイルパス>

CSVの各行の本文に、ユーザー辞書に登録済みの単語が含まれていれば、
その単語を「単独」で読ませた場合と「実際の文中」で読ませた場合の
アクセントを比較し、ズレていれば一覧に出す。
"""
import sys
import csv
import json
import urllib.request
import urllib.parse

VOICEVOX_URL = "http://127.0.0.1:50121"
SPEAKER_ID = 10002


def audio_query(text):
    url = f"{VOICEVOX_URL}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER_ID}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req) as res:
        return json.load(res)


def flatten_moras(query):
    """(mora_text, phrase_index, phrase_accent, phrase_mora_count) のリストを返す"""
    flat = []
    for pi, ap in enumerate(query.get("accent_phrases", [])):
        moras = ap["moras"]
        for m in moras:
            flat.append((m["text"], pi, ap.get("accent"), len(moras)))
    return flat


def get_user_dict():
    req = urllib.request.Request(f"{VOICEVOX_URL}/user_dict", method="GET")
    with urllib.request.urlopen(req) as res:
        return json.load(res)


def find_mora_span(flat_target_texts, flat_full):
    """flat_fullの中でflat_target_texts(モーラ文字列のリスト)と一致する連続区間を探す"""
    n, m = len(flat_full), len(flat_target_texts)
    for start in range(0, n - m + 1):
        if [flat_full[start + i][0] for i in range(m)] == flat_target_texts:
            return start
    return -1


def main(csv_path):
    words = get_user_dict()
    # surfaceでの検索を高速化 & 短い単語から先にマッチさせないよう長い順に
    entries = sorted(words.values(), key=lambda w: -len(w["surface"]))

    # 各単語のモーラ列を先に取得(単語自体のaccent_query結果からモーラ分割を借りる)
    word_cache = {}
    for w in entries:
        surface = w["surface"]
        if surface in word_cache:
            continue
        try:
            q = audio_query(surface)
        except Exception as e:
            continue
        flat = flatten_moras(q)
        if not flat:
            continue
        if len({pi for _, pi, _, _ in flat}) > 1:
            # 単語単体でも複数アクセント句に分かれる(長い複合語など)場合は
            # 単純比較の前提が崩れるため、このプロトタイプでは対象外にする
            continue
        # 比較は audio_query が返す accent フィールド同士で行う。
        # user_dict の accent_type(登録値)とは数値の意味が異なる場合がある
        # (例: 平板型は accent_type=0 だが、audio_queryのaccentはmora数と同じ値で返る)。
        word_cache[surface] = {
            "mora_texts": [t for t, _, _, _ in flat],
            "expected_accent": flat[0][2],
            "expected_mora_count": w["mora_count"],
        }

    mismatches = []
    checked = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 3:
                continue
            qid, genre, text = row[0], row[1], row[2]
            if not text or not text.strip():
                continue

            # この行に含まれる登録単語を探す(長い順なので部分文字列の誤マッチを避けやすい)
            matched_surfaces = [s for s in word_cache if s and s in text]
            if not matched_surfaces:
                continue

            try:
                full_q = audio_query(text)
            except Exception:
                continue
            flat_full = flatten_moras(full_q)
            checked += 1

            for surface in matched_surfaces:
                info = word_cache[surface]
                span_start = find_mora_span(info["mora_texts"], flat_full)
                if span_start == -1:
                    mismatches.append({
                        "qid": qid, "surface": surface, "issue": "モーラ列が文中に見つからない(表記/読みの不一致の疑い)",
                        "expected_accent": info["expected_accent"], "actual_accent": None,
                    })
                    continue

                span_len = len(info["mora_texts"])
                phrase_indices = {flat_full[span_start + i][1] for i in range(span_len)}
                if len(phrase_indices) > 1:
                    mismatches.append({
                        "qid": qid, "surface": surface, "issue": "文中で複数アクセント句にまたがっている(結合異常の疑い)",
                        "expected_accent": info["expected_accent"], "actual_accent": None,
                    })
                    continue

                pi = phrase_indices.pop()
                phrase_accent = flat_full[span_start][2]
                phrase_mora_count = flat_full[span_start][3]

                # 句の先頭インデックスを逆search
                phrase_start_idx = span_start
                while phrase_start_idx > 0 and flat_full[phrase_start_idx - 1][1] == pi:
                    phrase_start_idx -= 1
                prefix_extra = span_start - phrase_start_idx
                suffix_extra = phrase_mora_count - prefix_extra - span_len

                alone_accent = info["expected_accent"]
                alone_mora = info["expected_mora_count"]
                is_odaka_or_heiban = (alone_accent == alone_mora)

                is_bug = False
                if prefix_extra > 0:
                    # 前の語に取り込まれている = ほぼ確実に融合バグ
                    is_bug = True
                elif is_odaka_or_heiban:
                    # 尾高・平板型は、後ろに助詞がついても発音自体は変わらないが、
                    # accent値は句全体の拍数に合わせて機械的にズレるのが正常。
                    # (句頭からのズレなので) 期待値 = 句全体の拍数
                    is_bug = (phrase_accent != phrase_mora_count)
                else:
                    # 語中に本来の下がり目がある(頭高・中高)場合、後ろに何が付いても
                    # accent値は単語単体のときと変わらないのが正常。
                    is_bug = (phrase_accent != alone_accent)

                if is_bug:
                    parts = []
                    if prefix_extra > 0:
                        parts.append(f"前に{prefix_extra}拍分の語が結合")
                    if suffix_extra > 0:
                        parts.append(f"後ろに{suffix_extra}拍分の語が結合")
                    note = f"({', '.join(parts)})" if parts else ""
                    mismatches.append({
                        "qid": qid, "surface": surface,
                        "issue": f"アクセント位置が登録値と異なる{note}",
                        "expected_accent": alone_accent, "actual_accent": phrase_accent,
                    })

    print(f"チェック対象: {checked}行 / 登録単語 {len(word_cache)}件")
    print(f"不一致: {len(mismatches)}件\n")
    for m in mismatches:
        print(f"[{m['qid']}] 「{m['surface']}」 - {m['issue']} "
              f"(登録accent={m['expected_accent']} / 文中accent={m['actual_accent']})")


if __name__ == "__main__":
    main(sys.argv[1])
