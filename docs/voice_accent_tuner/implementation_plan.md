# 実装計画: 声でアクセント調整できる音声チューナーWebツール

## 概要
ブラウザからマイク録音を行い、ユーザーの声の抑揚を解析してVOICEVOX NemoのピッチパラメータにマッピングするWebツール。

## アーキテクチャ
- **バックエンド (Python)**:
  - 軽量Webサーバー ( /  or standard library based)
  - 音声解析: 純粋なPython + numpy/scipyによるYIN/オート相関ピッチ（F0）抽出
  - VOICEVOX Nemo () 連携
- **フロントエンド (HTML5 / Vanilla CSS / JS)**:
  - 美麗なダークモードUI
  - 問題番号（q0001〜q0300）の選択・CSV自動連動
  - リアルタイム録音・波形表示・ピッチ反映プレビュー
  - 1クリック上書き保存機能 (ffmpeg + EBU R128音量均一化自動適用)
