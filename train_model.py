import spacy
from spacy.training import Example

# **GiNZAベースのカスタムモデルを作成**
nlp = spacy.load("ja_ginza")
ner = nlp.get_pipe("ner")

# **除外する名前リスト**
EXCLUDED_NAMES = {
    "〇〇式": ["長谷川", "杉岡", "吉田", "田中", "佐藤"],
    "〇〇病": ["橋本", "川崎", "平山", "小柳・原田", "菊池", "瀬川", "木村", "網谷"],
    "〇〇法": ["柳原", "中西", "上川", "大塚", "吉松", "松代", "黒川", "森田", "Cant-Miwa"],
    "〇〇分類": ["正岡", "戸谷", "野口", "神谷", "福田", "山田", "大畑", "鈴木", "中村"]
}

# **カスタムNER学習用データセット**
TRAIN_DATA = [ 
    ("橋本病の研究が進んでいる", {"entities": [(0, 3, "MISC")]}) ,
    ("川崎病の新しい治療法", {"entities": [(0, 3, "MISC")]}) ,
    ("小柳・原田病は自己免疫疾患である", {"entities": [(0, 7, "MISC")]}) ,
    ("菊池病はリンパ節に関連する", {"entities": [(0, 3, "MISC")]}) ,
    ("鈴木教授が高安動脈炎の研究を行った", {"entities": [(0, 4, "Person")]}) ,
    ("山田先生が糖尿病の臨床試験を担当した", {"entities": [(0, 4, "Person")]}) ,
    ("長谷川式の評価方法が用いられる", {"entities": [(0, 4, "MISC")]}) ,
    ("杉岡式は有名な測定法である", {"entities": [(0, 3, "MISC")]}) ,
    ("吉田式の基準が改定された", {"entities": [(0, 3, "MISC")]}) ,
    ("田中式が採用されている", {"entities": [(0, 3, "MISC")]}) ,
    ("佐藤式の研究が進んでいる", {"entities": [(0, 3, "MISC")]}) ,
    ("平山病の原因が特定された", {"entities": [(0, 3, "MISC")]}) ,
    ("瀬川病の新薬が開発された", {"entities": [(0, 3, "MISC")]}) ,
    ("木村病は自己免疫疾患である", {"entities": [(0, 3, "MISC")]}) ,
    ("網谷病の診断が行われた", {"entities": [(0, 3, "MISC")]}) ,
    ("柳原法の適用範囲が広がる", {"entities": [(0, 3, "MISC")]}) ,
    ("中西法が新たに導入された", {"entities": [(0, 3, "MISC")]}) ,
    ("上川法の詳細が発表された", {"entities": [(0, 3, "MISC")]}) ,
    ("大塚法が正式に承認された", {"entities": [(0, 3, "MISC")]}) ,
    ("吉松法の改訂が行われた", {"entities": [(0, 3, "MISC")]}) ,
    ("黒川法が広く利用されている", {"entities": [(0, 3, "MISC")]}) ,
    ("正岡分類の変更が発表された", {"entities": [(0, 2, "MISC")]}) ,
    ("戸谷分類の適用が進んでいる", {"entities": [(0, 2, "MISC")]}) ,
    ("野口分類の基準が厳格化された", {"entities": [(0, 2, "MISC")]}) ,
    ("神谷分類の修正が行われた", {"entities": [(0, 2, "MISC")]}) ,
    ("福田分類の導入が検討されている", {"entities": [(0, 2, "MISC")]}) ,
    ("中村分類の新バージョンが公開された", {"entities": [(0, 2, "MISC")]}) ,
]

# **カスタムNERデータの追加**
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])  # MISC, Person を追加

# **トレーニングデータの作成**
examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in TRAIN_DATA]

# **学習**
optimizer = nlp.resume_training()
for i in range(10):  # エポック数
    for example in examples:
        nlp.update([example], drop=0.5, losses={})

# **学習済みモデルの保存**
nlp.to_disk("ja_ginza_custom")

print("カスタム学習済みモデル ja_ginza_custom を保存しました！")

