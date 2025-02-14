import streamlit as st
import pandas as pd
import spacy
import csv
import io
import re

# GiNZAをロード
nlp = spacy.load("ja_ginza")

# **Excelコピー時のデータをブロック単位で分割**
def parse_csv_text(text):
    text = text.strip()
    reader = csv.reader(io.StringIO(text))
    return [row[0] for row in reader if row]  # 1列目のデータのみを取得

# **追加の匿名化ルール（メール・電話番号・住所のマスキング）**
def advanced_anonymization(text):
    text = re.sub(r'\b\d{2,4}-\d{2,4}-\d{4}\b', '[匿_電話番号]', text)  # 電話番号
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[匿_メール]', text)  # メール
    text = re.sub(r'\b(東京都|大阪府|横浜市)[\w\d一-龥]+', '[匿_住所]', text)  # 住所（簡易）
    return text

# 匿名化関数（人名 + 追加ルール）
def anonymize_names(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["Person", "PERSON"]:
            text = text.replace(ent.text, "[匿_人名]")
    return advanced_anonymization(text)  # 追加の匿名化ルール適用

# StreamlitアプリのUI
st.title("医療テキストの匿名化ツール")

# テキストエリア
input_text = st.text_area("テキストを入力してください（Excelからのコピー可）", height=300)

# ボタンが押されたら処理を実行
if st.button("匿名化を実行"):
    if input_text.strip():  # 入力が空でないか確認
        original_texts = parse_csv_text(input_text)  # **CSVとして解析**
        processed_texts = [anonymize_names(text) for text in original_texts]

        # DataFrame に変換
        df = pd.DataFrame({"加工前のテキスト": original_texts, "加工後のテキスト": processed_texts})

        # **スクロール可能なDataFrameで表示（大量データ対応）**
        st.dataframe(df, height=600, use_container_width=True)

        # **Excelダウンロード用の処理**
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            writer.close()

        # **Excel ダウンロードボタン**
        st.download_button("Excelとしてダウンロード", buffer.getvalue(), file_name="anonymized_data.xlsx")

    else:
        st.warning("テキストを入力してください")

