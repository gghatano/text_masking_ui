import streamlit as st
import pandas as pd
import spacy
import csv
import io
import re

# **カスタム学習済み GiNZA モデルをロード**
try:
    nlp = spacy.load("ja_ginza_custom")  # 事前に学習したカスタムモデル
except OSError:
    st.error("カスタム学習済みモデルが見つかりません。`ja_ginza_custom` を事前に学習してください。")
    st.stop()

# **Excelコピー時のデータをブロック単位で分割**
def parse_csv_text(text):
    text = text.strip()
    reader = csv.reader(io.StringIO(text))
    return [row[0] for row in reader if row]  # 1列目のデータのみを取得

# **匿名化関数（人名のみマスキング、病名はそのまま）**
def anonymize_names(text):
    doc = nlp(text)

    # **エンティティ情報のデバッグ出力**
    print("----")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

    # **エンティティのリストを保持**
    entity_list = [(ent.text, ent.label_) for ent in doc.ents]

    # **人名エンティティをマスキング**
    for ent_text, ent_label in entity_list:
        if ent_label in ["Person", "PERSON"]:
            text = text.replace(ent_text, "[匿_人名]")

    return text

# StreamlitアプリのUI
st.title("医療テキストの匿名化ツール（カスタムモデル適用版）")

# **テキストエリア**
input_text = st.text_area("テキストを入力してください（Excelからのコピー可）", height=300)

# **ボタンが押されたら処理を実行**
if st.button("匿名化を実行"):
    if input_text.strip():  # **入力が空でないか確認**
        original_texts = parse_csv_text(input_text)  # **CSVとして解析**
        processed_texts = [anonymize_names(text) for text in original_texts]

        # **DataFrame に変換**
        df = pd.DataFrame({"加工前のテキスト": original_texts, "加工後のテキスト": processed_texts})

        # **折り返し表示用のCSS適用**
        st.markdown("""
            <style>
                table {
                    width: 100%;
                    table-layout: fixed;
                }
                th, td {
                    word-wrap: break-word;
                    white-space: pre-wrap;  /* 折り返し表示 */
                    text-align: left;
                }
            </style>
        """, unsafe_allow_html=True)

        # **HTMLテーブルとして表示**
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # **Excelダウンロード用の処理**
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
            writer.close()

        # **Excel ダウンロードボタン**
        st.download_button("Excelとしてダウンロード", buffer.getvalue(), file_name="anonymized_data.xlsx")

    else:
        st.warning("テキストを入力してください")

