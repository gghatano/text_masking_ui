import streamlit as st
import pandas as pd
import spacy
import csv
import io

# GiNZAをロード
nlp = spacy.load("ja_ginza")

# **CSVとしてExcelコピー内容を適切に解析**
def parse_csv_text(text):
    text = text.strip()
    reader = csv.reader(io.StringIO(text))
    return [row[0] for row in reader if row]  # 1列目のデータのみを取得

# 匿名化関数
def anonymize_names(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["Person", "PERSON"]:
            text = text.replace(ent.text, "[匿_人名]")
    return text

# StreamlitアプリのUI
st.title("医療テキストの匿名化ツール")

# テキストエリア
input_text = st.text_area("テキストを入力してください（Excelからのコピー可）", height=300)

# ボタンが押されたら処理を実行
if st.button("匿名化を実行"):
    if input_text.strip():  # 入力が空でないか確認
        original_texts = parse_csv_text(input_text)  # **CSVとして解析**
        processed_texts = [anonymize_names(text) for text in original_texts]

        # DataFrame に変換（改行を <br> に変換して HTML レンダリング）
        df = pd.DataFrame({
            "加工前のテキスト": [text.replace("\n", "<br>") for text in original_texts],
            "加工後のテキスト": [text.replace("\n", "<br>") for text in processed_texts]
        })

        # **HTML テーブルで表示**
        st.markdown(
            df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

    else:
        st.warning("テキストを入力してください")

